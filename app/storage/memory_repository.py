import sqlite3
from contextlib import closing, contextmanager
from datetime import datetime, timezone
from pathlib import Path

from app.memory.models import Memory, MemoryStatus
from app.storage.database import DEFAULT_DB_PATH, get_connection


_UNSET = object()


@contextmanager
def _connection_scope(
    db_path: Path | str,
    connection: sqlite3.Connection | None,
):
    if connection is not None:
        yield connection
    else:
        with closing(get_connection(db_path)) as conn, conn:
            yield conn


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
        memory.canonical_memory_id,
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
        canonical_memory_id=row["canonical_memory_id"],
    )


def insert(
    memory: Memory,
    db_path: Path | str = DEFAULT_DB_PATH,
    connection: sqlite3.Connection | None = None,
) -> None:
    with _connection_scope(db_path, connection) as conn:
        conn.execute(
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
                supersedes_id,
                canonical_memory_id
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            _memory_to_row(memory),
        )


def get(
    memory_id: str,
    db_path: Path | str = DEFAULT_DB_PATH,
    connection: sqlite3.Connection | None = None,
) -> Memory | None:
    with _connection_scope(db_path, connection) as conn:
        row = conn.execute(
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
    connection: sqlite3.Connection | None = None,
) -> list[Memory]:
    with _connection_scope(db_path, connection) as conn:
        rows = conn.execute(
            """
            SELECT *
            FROM memory
            WHERE memory_key = ?
              AND canonical_memory_id IS NULL
            ORDER BY valid_from ASC
            """,
            (memory_key,),
        ).fetchall()

    return [_row_to_memory(row) for row in rows]


def list_consolidated_into(
    canonical_memory_id: str,
    db_path: Path | str = DEFAULT_DB_PATH,
    connection: sqlite3.Connection | None = None,
) -> list[Memory]:
    with _connection_scope(db_path, connection) as conn:
        rows = conn.execute(
            """
            SELECT *
            FROM memory
            WHERE canonical_memory_id = ?
            ORDER BY recorded_at ASC, memory_id ASC
            """,
            (canonical_memory_id,),
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
        row = connection.execute(
            """
            SELECT *
            FROM memory
            WHERE memory_key = ?
              AND canonical_memory_id IS NULL
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


def update(
    memory_id: str,
    *,
    valid_to: datetime | None,
    status: MemoryStatus,
    canonical_memory_id: str | None | object = _UNSET,
    db_path: Path | str = DEFAULT_DB_PATH,
    connection: sqlite3.Connection | None = None,
) -> Memory | None:
    with _connection_scope(db_path, connection) as conn:
        existing = get(
            memory_id,
            connection=conn,
        )

        if existing is None:
            return None

        if canonical_memory_id is _UNSET:
            next_canonical_memory_id = existing.canonical_memory_id
        else:
            next_canonical_memory_id = canonical_memory_id

        updated = Memory(
            **{
                **existing.model_dump(),
                "valid_to": valid_to,
                "status": status,
                "canonical_memory_id": next_canonical_memory_id,
            }
        )

        conn.execute(
            """
            UPDATE memory
            SET valid_to = ?,
                status = ?,
                canonical_memory_id = ?
            WHERE memory_id = ?
            """,
            (
                (
                    _serialize_datetime(updated.valid_to)
                    if updated.valid_to is not None
                    else None
                ),
                updated.status.value,
                updated.canonical_memory_id,
                updated.memory_id,
            ),
        )

    return updated