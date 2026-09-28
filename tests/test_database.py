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
