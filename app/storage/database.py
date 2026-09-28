from contextlib import closing
from pathlib import Path
import sqlite3


DEFAULT_DB_PATH = Path("data/memory.db")


def get_connection(
    db_path: Path | str = DEFAULT_DB_PATH,
) -> sqlite3.Connection:
    if str(db_path) != ":memory:":
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(db_path)
    connection.row_factory = sqlite3.Row

    return connection


def initialize_database(
    db_path: Path | str = DEFAULT_DB_PATH,
) -> None:
    with closing(get_connection(db_path)) as connection, connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS memory (
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
                confidence REAL NOT NULL
                    CHECK (confidence BETWEEN 0 AND 1),
                status TEXT NOT NULL,
                supersedes_id TEXT,
                CHECK (
                    valid_to IS NULL
                    OR valid_to > valid_from
                )
            )
            """
        )

        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_memory_key_valid_from
            ON memory(memory_key, valid_from)
            """
        )
