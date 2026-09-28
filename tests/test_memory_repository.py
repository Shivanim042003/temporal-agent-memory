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
    list_by_key,
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


def test_list_by_key_returns_complete_timeline(database_path):
    node = make_memory(
        memory_id="mem_node",
        valid_from=datetime(2026, 1, 1, tzinfo=timezone.utc),
        valid_to=datetime(2026, 6, 1, tzinfo=timezone.utc),
        value="Node.js",
    )

    go = make_memory(
        memory_id="mem_go",
        valid_from=datetime(2026, 6, 1, tzinfo=timezone.utc),
        valid_to=datetime(2026, 9, 1, tzinfo=timezone.utc),
        value="Go",
        status=MemoryStatus.SUPERSEDED,
        supersedes_id="mem_node",
    )

    python = make_memory(
        memory_id="mem_python",
        valid_from=datetime(2026, 9, 1, tzinfo=timezone.utc),
        valid_to=None,
        value="Python",
        status=MemoryStatus.ACTIVE,
        supersedes_id="mem_go",
    )

    insert(node, database_path)
    insert(go, database_path)
    insert(python, database_path)

    timeline = list_by_key(
        "primary_backend_language",
        database_path,
    )

    assert len(timeline) == 3
    assert [memory.value for memory in timeline] == [
        "Node.js",
        "Go",
        "Python",
    ]


def test_list_by_key_orders_by_valid_from(database_path):
    python = make_memory(
        memory_id="mem_python",
        valid_from=datetime(2026, 9, 1, tzinfo=timezone.utc),
        valid_to=None,
        value="Python",
        status=MemoryStatus.ACTIVE,
    )

    node = make_memory(
        memory_id="mem_node",
        valid_from=datetime(2026, 1, 1, tzinfo=timezone.utc),
        valid_to=datetime(2026, 6, 1, tzinfo=timezone.utc),
        value="Node.js",
        status=MemoryStatus.HISTORICAL,
    )

    go = make_memory(
        memory_id="mem_go",
        valid_from=datetime(2026, 6, 1, tzinfo=timezone.utc),
        valid_to=datetime(2026, 9, 1, tzinfo=timezone.utc),
        value="Go",
        status=MemoryStatus.SUPERSEDED,
        supersedes_id="mem_node",
    )

    # Intentionally insert out of chronological order.
    insert(python, database_path)
    insert(node, database_path)
    insert(go, database_path)

    timeline = list_by_key(
        "primary_backend_language",
        database_path,
    )

    assert [memory.value for memory in timeline] == [
        "Node.js",
        "Go",
        "Python",
    ]


def test_list_by_key_excludes_other_memory_keys(database_path):
    backend = make_memory(
        memory_id="mem_backend",
        memory_key="primary_backend_language",
        value="Node.js",
        valid_to=None,
        status=MemoryStatus.ACTIVE,
    )

    db_memory = make_memory(
        memory_id="mem_database",
        memory_key="primary_database",
        value="PostgreSQL",
        valid_to=None,
        status=MemoryStatus.ACTIVE,
    )

    insert(backend, database_path)
    insert(db_memory, database_path)

    timeline = list_by_key(
        "primary_backend_language",
        database_path,
    )

    assert len(timeline) == 1
    assert timeline[0].value == "Node.js"


def test_list_by_key_unknown_key_returns_empty_list(database_path):
    timeline = list_by_key(
        "does_not_exist",
        database_path,
    )

    assert timeline == []