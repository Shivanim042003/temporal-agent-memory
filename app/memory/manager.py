from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from app.memory.models import (
    EvidenceType,
    Memory,
    MemoryStatus,
    MemoryType,
    SourceType,
    TimePrecision,
)
from app.storage.database import DEFAULT_DB_PATH, get_connection
from app.storage.memory_repository import (
    get as repo_get,
    get_at_time as repo_get_at_time,
    insert,
    list_by_key,
    update as repo_update,
)


class OverlapError(Exception):
    pass


class NoOpenMemoryError(Exception):
    pass


class MemoryNotFoundError(Exception):
    pass


class InvalidConsolidationError(Exception):
    pass


def _overlaps(
    a_from: datetime,
    a_to: datetime | None,
    b_from: datetime,
    b_to: datetime | None,
) -> bool:
    far_future = datetime.max.replace(tzinfo=timezone.utc)

    a_end = a_to if a_to is not None else far_future
    b_end = b_to if b_to is not None else far_future

    return (
        a_from < b_end
        and b_from < a_end
    )


def store(
    *,
    memory_key: str,
    subject: str,
    attribute: str,
    value: str,
    memory_type: MemoryType,
    valid_from: datetime,
    valid_to: datetime | None,
    precision: TimePrecision,
    source_type: SourceType,
    source_id: str,
    evidence_type: EvidenceType,
    confidence: float,
    supersedes_id: str | None = None,
    db_path: Path | str = DEFAULT_DB_PATH,
) -> Memory:
    status = (
        MemoryStatus.ACTIVE
        if valid_to is None
        else MemoryStatus.HISTORICAL
    )

    memory = Memory(
        memory_id=uuid4().hex,
        memory_key=memory_key,
        subject=subject,
        attribute=attribute,
        value=value,
        memory_type=memory_type,
        valid_from=valid_from,
        valid_to=valid_to,
        precision=precision,
        source_type=source_type,
        source_id=source_id,
        evidence_type=evidence_type,
        confidence=confidence,
        status=status,
        supersedes_id=supersedes_id,
    )

    connection = get_connection(db_path)

    try:
        connection.execute("BEGIN IMMEDIATE")

        existing_memories = list_by_key(
            memory_key,
            connection=connection,
        )

        for existing in existing_memories:
            if _overlaps(
                existing.valid_from,
                existing.valid_to,
                memory.valid_from,
                memory.valid_to,
            ):
                raise OverlapError(
                    f"Memory interval overlaps existing memory "
                    f"{existing.memory_id}: "
                    f"[{existing.valid_from}, {existing.valid_to})"
                )

        insert(
            memory,
            connection=connection,
        )

        connection.commit()

        return memory

    except Exception:
        connection.rollback()
        raise

    finally:
        connection.close()


def supersede(
    *,
    memory_key: str,
    subject: str,
    attribute: str,
    value: str,
    memory_type: MemoryType,
    valid_from: datetime,
    precision: TimePrecision,
    source_type: SourceType,
    source_id: str,
    evidence_type: EvidenceType,
    confidence: float,
    db_path: Path | str = DEFAULT_DB_PATH,
) -> Memory:
    if valid_from.tzinfo is None:
        raise ValueError("datetime must be timezone-aware")

    valid_from = valid_from.astimezone(timezone.utc)

    connection = get_connection(db_path)

    try:
        connection.execute("BEGIN IMMEDIATE")

        timeline = list_by_key(
            memory_key,
            connection=connection,
        )

        open_memories = [
            memory
            for memory in timeline
            if memory.valid_to is None
        ]

        if not open_memories:
            raise NoOpenMemoryError(
                f"No open memory exists for key '{memory_key}'"
            )

        if len(open_memories) > 1:
            raise OverlapError(
                f"Multiple open memories exist for key "
                f"'{memory_key}'"
            )

        current = open_memories[0]

        if valid_from <= current.valid_from:
            raise ValueError(
                "new valid_from must be strictly after "
                "the current memory's valid_from"
            )

        new_memory = Memory(
            memory_id=uuid4().hex,
            memory_key=memory_key,
            subject=subject,
            attribute=attribute,
            value=value,
            memory_type=memory_type,
            valid_from=valid_from,
            valid_to=None,
            precision=precision,
            source_type=source_type,
            source_id=source_id,
            evidence_type=evidence_type,
            confidence=confidence,
            status=MemoryStatus.ACTIVE,
            supersedes_id=current.memory_id,
        )

        closed = repo_update(
            current.memory_id,
            valid_to=new_memory.valid_from,
            status=MemoryStatus.SUPERSEDED,
            connection=connection,
        )

        if closed is None:
            raise NoOpenMemoryError(
                f"Memory {current.memory_id} disappeared"
            )

        insert(
            new_memory,
            connection=connection,
        )

        connection.commit()

        return new_memory

    except Exception:
        connection.rollback()
        raise

    finally:
        connection.close()


def consolidate(
    memory_ids: list[str],
    *,
    canonical_memory_id: str | None = None,
    db_path: Path | str = DEFAULT_DB_PATH,
) -> Memory:
    if len(memory_ids) < 2:
        raise InvalidConsolidationError(
            "consolidate() requires at least two memories"
        )

    if len(set(memory_ids)) != len(memory_ids):
        raise InvalidConsolidationError(
            "consolidate() received duplicate memory_ids"
        )

    connection = get_connection(db_path)

    try:
        connection.execute("BEGIN IMMEDIATE")

        candidates: list[Memory] = []

        for memory_id in memory_ids:
            memory = repo_get(
                memory_id,
                connection=connection,
            )

            if memory is None:
                raise MemoryNotFoundError(
                    f"Memory {memory_id} does not exist"
                )

            if memory.status == MemoryStatus.CONSOLIDATED:
                raise InvalidConsolidationError(
                    f"Memory {memory_id} is already consolidated "
                    f"into {memory.canonical_memory_id}"
                )

            candidates.append(memory)

        first = candidates[0]

        for other in candidates[1:]:
            if (
                other.memory_key != first.memory_key
                or other.subject != first.subject
                or other.attribute != first.attribute
                or other.value != first.value
                or other.valid_from != first.valid_from
                or other.valid_to != first.valid_to
            ):
                raise InvalidConsolidationError(
                    "all candidates must share memory_key, subject, "
                    "attribute, value, valid_from, and valid_to"
                )

        if canonical_memory_id is not None:
            canonical = next(
                (
                    memory
                    for memory in candidates
                    if memory.memory_id == canonical_memory_id
                ),
                None,
            )

            if canonical is None:
                raise InvalidConsolidationError(
                    f"canonical_memory_id {canonical_memory_id} "
                    f"is not in the candidate set"
                )
        else:
            canonical = min(
                candidates,
                key=lambda memory: memory.recorded_at,
            )

        for candidate in candidates:
            if candidate.memory_id == canonical.memory_id:
                continue

            updated = repo_update(
                candidate.memory_id,
                valid_to=candidate.valid_to,
                status=MemoryStatus.CONSOLIDATED,
                canonical_memory_id=canonical.memory_id,
                connection=connection,
            )

            if updated is None:
                raise MemoryNotFoundError(
                    f"Memory {candidate.memory_id} disappeared "
                    "during consolidation"
                )

        connection.commit()

        return canonical

    except Exception:
        connection.rollback()
        raise

    finally:
        connection.close()


def get_at_time(
    memory_key: str,
    at: datetime,
    db_path: Path | str = DEFAULT_DB_PATH,
) -> Memory | None:
    return repo_get_at_time(
        memory_key,
        at,
        db_path,
    )


def get_current(
    memory_key: str,
    db_path: Path | str = DEFAULT_DB_PATH,
) -> Memory | None:
    timeline = list_by_key(memory_key, db_path)

    for memory in timeline:
        if memory.valid_to is None:
            return memory

    return None


def get_range(
    memory_key: str,
    start: datetime,
    end: datetime,
    db_path: Path | str = DEFAULT_DB_PATH,
) -> list[Memory]:
    if start.tzinfo is None or end.tzinfo is None:
        raise ValueError("datetime must be timezone-aware")

    start = start.astimezone(timezone.utc)
    end = end.astimezone(timezone.utc)

    if end <= start:
        raise ValueError("end must be strictly after start")

    timeline = list_by_key(memory_key, db_path)

    return [
        memory
        for memory in timeline
        if _overlaps(
            memory.valid_from,
            memory.valid_to,
            start,
            end,
        )
    ]


def get_transition(
    memory_key: str,
    from_value: str,
    to_value: str,
    db_path: Path | str = DEFAULT_DB_PATH,
) -> datetime | None:
    timeline = list_by_key(memory_key, db_path)
    by_id = {
        memory.memory_id: memory
        for memory in timeline
    }

    for memory in timeline:
        if memory.value != to_value:
            continue

        if memory.supersedes_id is None:
            continue

        previous = by_id.get(memory.supersedes_id)

        if (
            previous is not None
            and previous.value == from_value
        ):
            return memory.valid_from

    return None