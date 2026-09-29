import sqlite3

from app.storage.database import get_connection, initialize_database


def test_initialize_database_creates_memory_table(tmp_path):
    database_path = tmp_path / "test_memory.db"

    initialize_database(database_path)

    with get_connection(database_path) as connection:
        table = connection.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
              AND name = 'memory'
            """
        ).fetchone()

    assert table is not None
    assert table["name"] == "memory"


def test_memory_table_has_expected_columns(tmp_path):
    database_path = tmp_path / "test_memory.db"

    initialize_database(database_path)

    with get_connection(database_path) as connection:
        columns = connection.execute(
            "PRAGMA table_info(memory)"
        ).fetchall()

    column_names = [column["name"] for column in columns]

    assert column_names == [
        "memory_id",
        "memory_key",
        "subject",
        "attribute",
        "value",
        "memory_type",
        "valid_from",
        "valid_to",
        "precision",
        "recorded_at",
        "source_type",
        "source_id",
        "evidence_type",
        "confidence",
        "status",
        "supersedes_id",
        "canonical_memory_id",
    ]


def test_memory_id_is_primary_key(tmp_path):
    database_path = tmp_path / "test_memory.db"

    initialize_database(database_path)

    with get_connection(database_path) as connection:
        columns = connection.execute(
            "PRAGMA table_info(memory)"
        ).fetchall()

    memory_id_column = next(
        column
        for column in columns
        if column["name"] == "memory_id"
    )

    assert memory_id_column["pk"] == 1


def test_composite_memory_index_is_created(tmp_path):
    database_path = tmp_path / "test_memory.db"

    initialize_database(database_path)

    with get_connection(database_path) as connection:
        indexes = connection.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'index'
              AND name = 'idx_memory_key_valid_from'
            """
        ).fetchall()

    assert len(indexes) == 1


def test_get_connection_returns_sqlite_rows(tmp_path):
    database_path = tmp_path / "test_memory.db"

    initialize_database(database_path)

    with get_connection(database_path) as connection:
        row = connection.execute(
            "SELECT 1 AS value"
        ).fetchone()

    assert isinstance(row, sqlite3.Row)
    assert row["value"] == 1


def test_initialize_database_migrates_existing_database(tmp_path):
    database_path = tmp_path / "test_memory.db"

    with get_connection(database_path) as connection:
        connection.execute(
            """
            CREATE TABLE memory (
                memory_id TEXT PRIMARY KEY,
                memory_key TEXT NOT NULL,
                subject TEXT NOT NULL,
                attribute TEXT NOT NULL,
                value TEXT NOT NULL,
                memory_type TEXT NOT NULL,
                valid_from TEXT NOT NULL,
                valid_to TEXT,
                precision TEXT NOT NULL,
                recorded_at TEXT NOT NULL,
                source_type TEXT NOT NULL,
                source_id TEXT NOT NULL,
                evidence_type TEXT NOT NULL,
                confidence REAL NOT NULL,
                status TEXT NOT NULL,
                supersedes_id TEXT
            )
            """
        )

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
            (
                "legacy_memory",
                "primary_backend_language",
                "user",
                "uses",
                "Node.js",
                "skill",
                "2026-01-01T00:00:00+00:00",
                None,
                "day",
                "2026-01-15T00:00:00+00:00",
                "conversation",
                "conversation_001",
                "explicit",
                0.98,
                "active",
                None,
            ),
        )

        connection.commit()

    initialize_database(database_path)

    with get_connection(database_path) as connection:
        columns = connection.execute(
            "PRAGMA table_info(memory)"
        ).fetchall()

        column_names = [column["name"] for column in columns]

        row = connection.execute(
            """
            SELECT memory_id, value, canonical_memory_id
            FROM memory
            WHERE memory_id = ?
            """,
            ("legacy_memory",),
        ).fetchone()

    assert "canonical_memory_id" in column_names
    assert row["memory_id"] == "legacy_memory"
    assert row["value"] == "Node.js"
    assert row["canonical_memory_id"] is None