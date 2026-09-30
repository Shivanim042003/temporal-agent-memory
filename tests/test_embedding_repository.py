from pathlib import Path
import sqlite3

import pytest

from app.storage.database import initialize_database
from app.storage.embedding_repository import (
    delete,
    get,
    initialize_embedding_storage,
    upsert,
)


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    path = tmp_path / "memory.db"

    initialize_database(path)
    initialize_embedding_storage(path)

    with sqlite3.connect(path) as connection:
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
                supersedes_id,
                canonical_memory_id
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "mem_1",
                "user.programming_language",
                "user",
                "programming_language",
                "Python",
                "skill",
                "2026-01-01T00:00:00+00:00",
                None,
                "exact",
                "2026-01-01T00:00:00+00:00",
                "conversation",
                "conversation_1",
                "explicit",
                1.0,
                "active",
                None,
                None,
            ),
        )

    return path


def test_initialize_embedding_storage_creates_table(db_path: Path):
    with sqlite3.connect(db_path) as connection:
        row = connection.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
              AND name = 'memory_embedding'
            """
        ).fetchone()

    assert row is not None
    assert row[0] == "memory_embedding"


def test_upsert_and_get_embedding(db_path: Path):
    embedding = [0.1, 0.2, 0.3, 0.4]

    upsert(
        memory_id="mem_1",
        embedding=embedding,
        model_name="test-model",
        db_path=db_path,
    )

    result = get(
        memory_id="mem_1",
        db_path=db_path,
    )

    assert result is not None
    assert result["memory_id"] == "mem_1"
    assert result["model_name"] == "test-model"
    assert result["dimension"] == 4
    assert result["embedding"] == pytest.approx(embedding)


def test_get_returns_none_for_missing_embedding(db_path: Path):
    result = get(
        memory_id="does-not-exist",
        db_path=db_path,
    )

    assert result is None


def test_upsert_replaces_existing_embedding(db_path: Path):
    upsert(
        memory_id="mem_1",
        embedding=[0.1, 0.2],
        model_name="model-v1",
        db_path=db_path,
    )

    upsert(
        memory_id="mem_1",
        embedding=[0.3, 0.4, 0.5],
        model_name="model-v2",
        db_path=db_path,
    )

    result = get(
        memory_id="mem_1",
        db_path=db_path,
    )

    assert result is not None
    assert result["model_name"] == "model-v2"
    assert result["dimension"] == 3
    assert result["embedding"] == pytest.approx(
        [0.3, 0.4, 0.5]
    )


def test_delete_existing_embedding(db_path: Path):
    upsert(
        memory_id="mem_1",
        embedding=[0.1, 0.2],
        model_name="test-model",
        db_path=db_path,
    )

    deleted = delete(
        memory_id="mem_1",
        db_path=db_path,
    )

    assert deleted is True
    assert get(
        memory_id="mem_1",
        db_path=db_path,
    ) is None


def test_delete_missing_embedding_returns_false(db_path: Path):
    deleted = delete(
        memory_id="does-not-exist",
        db_path=db_path,
    )

    assert deleted is False


def test_upsert_rejects_empty_embedding(db_path: Path):
    with pytest.raises(
        ValueError,
        match="embedding must not be empty",
    ):
        upsert(
            memory_id="mem_1",
            embedding=[],
            model_name="test-model",
            db_path=db_path,
        )


def test_upsert_rejects_non_finite_embedding(db_path: Path):
    with pytest.raises(
        ValueError,
        match="embedding must contain only finite values",
    ):
        upsert(
            memory_id="mem_1",
            embedding=[0.1, float("nan")],
            model_name="test-model",
            db_path=db_path,
        )