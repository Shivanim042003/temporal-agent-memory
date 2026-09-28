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
    get_at_time as repo_get_at_time,
    insert,
    list_by_key,
    update,
)


class OverlapError(Exception):
    pass


class NoOpenMemoryError(Exception):
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

    # Check-then-insert is not atomic yet.
    # A later step will move this into one transaction.
    existing_memories = list_by_key(
        memory_key,
        db_path,
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

    insert(memory, db_path)

    return memory


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

        closed = update(
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