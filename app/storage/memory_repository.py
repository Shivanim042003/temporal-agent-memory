from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

from app.memory.models import Memory
from app.storage.database import DEFAULT_DB_PATH, get_connection


def _serialize_datetime(value: datetime) -> str:
    if value.tzinfo is None:
        raise ValueError("datetime must be timezone-aware")

    return value.astimezone(timezone.utc).isoformat(timespec="microseconds")


def _deserialize_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _memory_to_row(memory: Memory) -> tuple:
    return (
        memory.memory_id,
        memory.memory_key,
        memory.subject,
        memory.attribute,
        memory.value,
        memory.memory_type.value,
        _serialize_datetime(memory.valid_from),
        (
            _serialize_datetime(memory.valid_to)
            if memory.valid_to is not None
            else None
        ),
        memory.precision.value,
        _serialize_datetime(memory.recorded_at),
        memory.source_type.value,
        memory.source_id,
        memory.evidence_type.value,
        memory.confidence,
        memory.status.value,
        memory.supersedes_id,
    )


def _row_to_memory(row) -> Memory:
    return Memory(
        memory_id=row["memory_id"],
        memory_key=row["memory_key"],
        subject=row["subject"],
        attribute=row["attribute"],
        value=row["value"],
        memory_type=row["memory_type"],
        valid_from=_deserialize_datetime(row["valid_from"]),
        valid_to=(
            _deserialize_datetime(row["valid_to"])
            if row["valid_to"] is not None
            else None
        ),
        precision=row["precision"],
        recorded_at=_deserialize_datetime(row["recorded_at"]),
        source_type=row["source_type"],
        source_id=row["source_id"],
        evidence_type=row["evidence_type"],
        confidence=row["confidence"],
        status=row["status"],
        supersedes_id=row["supersedes_id"],
    )


def insert(
    memory: Memory,
    db_path: Path | str = DEFAULT_DB_PATH,
) -> None:
    with closing(get_connection(db_path)) as connection, connection:
        connection.execute(
            """
            INSERT INTO memory (
                memory_id,
                memory_key,
                subject,
                attribute,
                value,
                memory_type,
                valid_from,
                valid_to,
                precision,
                recorded_at,
                source_type,
                source_id,
                evidence_type,
                confidence,
                status,
                supersedes_id
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            _memory_to_row(memory),
        )


def get(
    memory_id: str,
    db_path: Path | str = DEFAULT_DB_PATH,
) -> Memory | None:
    with closing(get_connection(db_path)) as connection:
        row = connection.execute(
            """
            SELECT *
            FROM memory
            WHERE memory_id = ?
            """,
            (memory_id,),
        ).fetchone()

    if row is None:
        return None

    return _row_to_memory(row)


def list_by_key(
    memory_key: str,
    db_path: Path | str = DEFAULT_DB_PATH,
) -> list[Memory]:
    with closing(get_connection(db_path)) as connection:
        rows = connection.execute(
            """
            SELECT *
            FROM memory
            WHERE memory_key = ?
            ORDER BY valid_from ASC
            """,
            (memory_key,),
        ).fetchall()

    return [_row_to_memory(row) for row in rows]


def get_at_time(
    memory_key: str,
    at: datetime,
    db_path: Path | str = DEFAULT_DB_PATH,
) -> Memory | None:
    if at.tzinfo is None:
        raise ValueError("datetime must be timezone-aware")

    query_time = _serialize_datetime(at)

    with closing(get_connection(db_path)) as connection:
        # LIMIT 1 is temporary. Later, Memory Manager validation
        # will prevent overlapping intervals for the same memory_key.
        row = connection.execute(
            """
            SELECT *
            FROM memory
            WHERE memory_key = ?
              AND valid_from <= ?
              AND (valid_to IS NULL OR ? < valid_to)
            ORDER BY valid_from DESC
            LIMIT 1
            """,
            (memory_key, query_time, query_time),
        ).fetchone()

    if row is None:
        return None

    return _row_to_memory(row)