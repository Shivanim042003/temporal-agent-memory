import sqlite3
from datetime import datetime, timezone

import pytest

from app.memory.models import (
    EvidenceType,
    Memory,
    MemoryStatus,
    MemoryType,
    SourceType,
    TimePrecision,
)
from app.storage.database import initialize_database
from app.storage.memory_repository import (
    get,
    get_at_time,
    insert,
)


def make_memory(**overrides):
    data = {
        "memory_id": "mem_001",
        "memory_key": "primary_backend_language",
        "subject": "user",
        "attribute": "uses",
        "value": "Node.js",
        "memory_type": MemoryType.SKILL,
        "valid_from": datetime(2026, 1, 1, tzinfo=timezone.utc),
        "valid_to": datetime(2026, 6, 1, tzinfo=timezone.utc),
        "recorded_at": datetime(2026, 1, 15, tzinfo=timezone.utc),
        "precision": TimePrecision.DAY,
        "source_type": SourceType.CONVERSATION,
        "source_id": "conversation_001",
        "evidence_type": EvidenceType.EXPLICIT,
        "confidence": 0.98,
        "status": MemoryStatus.HISTORICAL,
        "supersedes_id": None,
    }

    data.update(overrides)
    return Memory(**data)


@pytest.fixture
def database_path(tmp_path):
    path = tmp_path / "test.db"
    initialize_database(path)
    return path


def test_insert_get_round_trip_preserves_memory(database_path):
    memory = make_memory(
        supersedes_id="mem_previous",
    )

    insert(memory, database_path)

    retrieved = get(memory.memory_id, database_path)

    assert retrieved == memory
    assert retrieved.memory_type == MemoryType.SKILL
    assert retrieved.status == MemoryStatus.HISTORICAL
    assert retrieved.evidence_type == EvidenceType.EXPLICIT
    assert retrieved.source_type == SourceType.CONVERSATION
    assert retrieved.supersedes_id == "mem_previous"
    assert retrieved.valid_from == memory.valid_from
    assert retrieved.valid_to == memory.valid_to
    assert retrieved.recorded_at == memory.recorded_at


def test_march_returns_node_js(database_path):
    node = make_memory(
        memory_id="mem_node",
        valid_from=datetime(2026, 1, 1, tzinfo=timezone.utc),
        valid_to=datetime(2026, 6, 1, tzinfo=timezone.utc),
        value="Node.js",
    )

    go = make_memory(
        memory_id="mem_go",
        valid_from=datetime(2026, 6, 1, tzinfo=timezone.utc),
        valid_to=None,
        value="Go",
        status=MemoryStatus.ACTIVE,
        supersedes_id="mem_node",
    )

    insert(node, database_path)
    insert(go, database_path)

    retrieved = get_at_time(
        "primary_backend_language",
        datetime(2026, 3, 1, tzinfo=timezone.utc),
        database_path,
    )

    assert retrieved is not None
    assert retrieved.value == "Node.js"
    assert retrieved.memory_id == "mem_node"


def test_july_returns_go(database_path):
    node = make_memory(
        memory_id="mem_node",
        valid_from=datetime(2026, 1, 1, tzinfo=timezone.utc),
        valid_to=datetime(2026, 6, 1, tzinfo=timezone.utc),
        value="Node.js",
    )

    go = make_memory(
        memory_id="mem_go",
        valid_from=datetime(2026, 6, 1, tzinfo=timezone.utc),
        valid_to=None,
        value="Go",
        status=MemoryStatus.ACTIVE,
        supersedes_id="mem_node",
    )

    insert(node, database_path)
    insert(go, database_path)

    retrieved = get_at_time(
        "primary_backend_language",
        datetime(2026, 7, 1, tzinfo=timezone.utc),
        database_path,
    )

    assert retrieved is not None
    assert retrieved.value == "Go"
    assert retrieved.memory_id == "mem_go"


def test_june_first_boundary_returns_go(database_path):
    node = make_memory(
        memory_id="mem_node",
        valid_from=datetime(2026, 1, 1, tzinfo=timezone.utc),
        valid_to=datetime(2026, 6, 1, tzinfo=timezone.utc),
        value="Node.js",
    )

    go = make_memory(
        memory_id="mem_go",
        valid_from=datetime(2026, 6, 1, tzinfo=timezone.utc),
        valid_to=None,
        value="Go",
        status=MemoryStatus.ACTIVE,
        supersedes_id="mem_node",
    )

    insert(node, database_path)
    insert(go, database_path)

    retrieved = get_at_time(
        "primary_backend_language",
        datetime(2026, 6, 1, tzinfo=timezone.utc),
        database_path,
    )

    assert retrieved is not None
    assert retrieved.value == "Go"
    assert retrieved.memory_id == "mem_go"


def test_time_before_first_memory_returns_none(database_path):
    node = make_memory(
        memory_id="mem_node",
        valid_from=datetime(2026, 1, 1, tzinfo=timezone.utc),
        valid_to=None,
        status=MemoryStatus.ACTIVE,
    )

    insert(node, database_path)

    retrieved = get_at_time(
        "primary_backend_language",
        datetime(2025, 12, 31, tzinfo=timezone.utc),
        database_path,
    )

    assert retrieved is None


def test_naive_at_raises_value_error(database_path):
    with pytest.raises(ValueError, match="timezone-aware"):
        get_at_time(
            "primary_backend_language",
            datetime(2026, 3, 1),
            database_path,
        )


def test_duplicate_memory_id_raises_integrity_error(database_path):
    memory = make_memory()

    insert(memory, database_path)

    with pytest.raises(sqlite3.IntegrityError):
        insert(memory, database_path)


def test_unknown_memory_key_returns_none(database_path):
    retrieved = get_at_time(
        "unknown_memory_key",
        datetime(2026, 3, 1, tzinfo=timezone.utc),
        database_path,
    )

    assert retrieved is None
