import sqlite3
from contextlib import closing, contextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from app.storage.database import DEFAULT_DB_PATH, get_connection


EMBEDDING_TABLE = "memory_embedding"


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


def initialize_embedding_storage(
    db_path: Path | str = DEFAULT_DB_PATH,
) -> None:
    with closing(get_connection(db_path)) as connection, connection:
        connection.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {EMBEDDING_TABLE} (
                memory_id TEXT PRIMARY KEY,
                model_name TEXT NOT NULL,
                dimension INTEGER NOT NULL CHECK (dimension > 0),
                embedding BLOB NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (memory_id)
                    REFERENCES memory(memory_id)
            )
            """
        )

        connection.execute(
            f"""
            CREATE INDEX IF NOT EXISTS idx_memory_embedding_model
            ON {EMBEDDING_TABLE}(model_name)
            """
        )


def _serialize_embedding(embedding: list[float]) -> tuple[bytes, int]:
    if not embedding:
        raise ValueError("embedding must not be empty")

    array = np.asarray(embedding, dtype=np.float32)

    if array.ndim != 1:
        raise ValueError("embedding must be one-dimensional")

    if not np.isfinite(array).all():
        raise ValueError("embedding must contain only finite values")

    return array.tobytes(), int(array.shape[0])


def _deserialize_embedding(
    embedding: bytes,
    dimension: int,
) -> list[float]:
    array = np.frombuffer(
        embedding,
        dtype=np.float32,
    )

    if len(array) != dimension:
        raise ValueError(
            "stored embedding dimension does not match metadata"
        )

    return array.tolist()


def upsert(
    memory_id: str,
    embedding: list[float],
    model_name: str,
    db_path: Path | str = DEFAULT_DB_PATH,
    connection: sqlite3.Connection | None = None,
) -> None:
    if not memory_id:
        raise ValueError("memory_id must not be empty")

    if not model_name:
        raise ValueError("model_name must not be empty")

    serialized, dimension = _serialize_embedding(embedding)

    created_at = datetime.now(timezone.utc).isoformat(
        timespec="microseconds"
    )

    with _connection_scope(db_path, connection) as conn:
        conn.execute(
            f"""
            INSERT INTO {EMBEDDING_TABLE} (
                memory_id,
                model_name,
                dimension,
                embedding,
                created_at
            )
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(memory_id)
            DO UPDATE SET
                model_name = excluded.model_name,
                dimension = excluded.dimension,
                embedding = excluded.embedding,
                created_at = excluded.created_at
            """,
            (
                memory_id,
                model_name,
                dimension,
                serialized,
                created_at,
            ),
        )


def get(
    memory_id: str,
    db_path: Path | str = DEFAULT_DB_PATH,
    connection: sqlite3.Connection | None = None,
) -> dict | None:
    with _connection_scope(db_path, connection) as conn:
        row = conn.execute(
            f"""
            SELECT
                memory_id,
                model_name,
                dimension,
                embedding,
                created_at
            FROM {EMBEDDING_TABLE}
            WHERE memory_id = ?
            """,
            (memory_id,),
        ).fetchone()

    if row is None:
        return None

    return {
        "memory_id": row["memory_id"],
        "model_name": row["model_name"],
        "dimension": row["dimension"],
        "embedding": _deserialize_embedding(
            row["embedding"],
            row["dimension"],
        ),
        "created_at": datetime.fromisoformat(row["created_at"]),
    }


def delete(
    memory_id: str,
    db_path: Path | str = DEFAULT_DB_PATH,
    connection: sqlite3.Connection | None = None,
) -> bool:
    with _connection_scope(db_path, connection) as conn:
        cursor = conn.execute(
            f"""
            DELETE FROM {EMBEDDING_TABLE}
            WHERE memory_id = ?
            """,
            (memory_id,),
        )

        return cursor.rowcount > 0