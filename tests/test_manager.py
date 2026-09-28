from datetime import datetime, timedelta, timezone

import pytest

from app.memory.manager import (
    OverlapError,
    _overlaps,
    get_at_time,
    store,
)
from app.memory.models import (
    EvidenceType,
    MemoryStatus,
    MemoryType,
    SourceType,
    TimePrecision,
)
from app.storage.database import initialize_database
from app.storage.memory_repository import list_by_key


@pytest.fixture
def database_path(tmp_path):
    path = tmp_path / "test.db"
    initialize_database(path)
    return path


def make_store_kwargs(**overrides):
    data = {
        "memory_key": "primary_backend_language",
        "subject": "user",
        "attribute": "uses",
        "value": "Node.js",
        "memory_type": MemoryType.SKILL,
        "valid_from": datetime(2026, 1, 1, tzinfo=timezone.utc),
        "valid_to": datetime(2026, 6, 1, tzinfo=timezone.utc),
        "precision": TimePrecision.DAY,
        "source_type": SourceType.CONVERSATION,
        "source_id": "conversation_001",
        "evidence_type": EvidenceType.EXPLICIT,
        "confidence": 0.98,
    }

    data.update(overrides)
    return data


def test_adjacent_intervals_do_not_overlap():
    june = datetime(2026, 6, 1, tzinfo=timezone.utc)

    assert _overlaps(
        datetime(2026, 1, 1, tzinfo=timezone.utc),
        june,
        june,
        None,
    ) is False


def test_overlapping_intervals_overlap():
    assert _overlaps(
        datetime(2026, 1, 1, tzinfo=timezone.utc),
        datetime(2026, 6, 1, tzinfo=timezone.utc),
        datetime(2026, 5, 1, tzinfo=timezone.utc),
        None,
    ) is True


def test_two_open_ended_intervals_overlap():
    assert _overlaps(
        datetime(2026, 1, 1, tzinfo=timezone.utc),
        None,
        datetime(2026, 6, 1, tzinfo=timezone.utc),
        None,
    ) is True


def test_non_overlapping_historical_intervals():
    assert _overlaps(
        datetime(2026, 1, 1, tzinfo=timezone.utc),
        datetime(2026, 3, 1, tzinfo=timezone.utc),
        datetime(2026, 3, 1, tzinfo=timezone.utc),
        datetime(2026, 6, 1, tzinfo=timezone.utc),
    ) is False


def test_inner_interval_overlaps_outer_interval():
    assert _overlaps(
        datetime(2026, 1, 1, tzinfo=timezone.utc),
        datetime(2026, 12, 1, tzinfo=timezone.utc),
        datetime(2026, 3, 1, tzinfo=timezone.utc),
        datetime(2026, 6, 1, tzinfo=timezone.utc),
    ) is True


def test_one_microsecond_overlap():
    june = datetime(2026, 6, 1, tzinfo=timezone.utc)

    assert _overlaps(
        datetime(2026, 1, 1, tzinfo=timezone.utc),
        june,
        june - timedelta(microseconds=1),
        None,
    ) is True


def test_store_node_js_then_go(database_path):
    node = store(
        **make_store_kwargs(
            value="Node.js",
            valid_from=datetime(2026, 1, 1, tzinfo=timezone.utc),
            valid_to=datetime(2026, 6, 1, tzinfo=timezone.utc),
            source_id="conversation_001",
        ),
        db_path=database_path,
    )

    go = store(
        **make_store_kwargs(
            value="Go",
            valid_from=datetime(2026, 6, 1, tzinfo=timezone.utc),
            valid_to=None,
            source_id="conversation_002",
            supersedes_id=node.memory_id,
        ),
        db_path=database_path,
    )

    timeline = list_by_key(
        "primary_backend_language",
        database_path,
    )

    assert len(timeline) == 2
    assert timeline[0].value == "Node.js"
    assert timeline[1].value == "Go"

    march = get_at_time(
        "primary_backend_language",
        datetime(2026, 3, 1, tzinfo=timezone.utc),
        database_path,
    )

    july = get_at_time(
        "primary_backend_language",
        datetime(2026, 7, 1, tzinfo=timezone.utc),
        database_path,
    )

    assert march is not None
    assert march.value == "Node.js"

    assert july is not None
    assert july.value == "Go"


def test_closed_interval_gets_historical_status(database_path):
    memory = store(
        **make_store_kwargs(
            valid_to=datetime(2026, 6, 1, tzinfo=timezone.utc),
        ),
        db_path=database_path,
    )

    assert memory.status == MemoryStatus.HISTORICAL


def test_open_interval_gets_active_status(database_path):
    memory = store(
        **make_store_kwargs(
            valid_to=None,
        ),
        db_path=database_path,
    )

    assert memory.status == MemoryStatus.ACTIVE


def test_overlapping_interval_raises_overlap_error(database_path):
    node = store(
        **make_store_kwargs(
            valid_to=datetime(2026, 6, 1, tzinfo=timezone.utc),
        ),
        db_path=database_path,
    )

    with pytest.raises(
        OverlapError,
        match=node.memory_id,
    ):
        store(
            **make_store_kwargs(
                value="Python",
                valid_from=datetime(2026, 5, 1, tzinfo=timezone.utc),
                valid_to=None,
                source_id="conversation_002",
            ),
            db_path=database_path,
        )


def test_two_open_ended_memories_for_same_key_raise_overlap_error(
    database_path,
):
    store(
        **make_store_kwargs(
            valid_to=None,
        ),
        db_path=database_path,
    )

    with pytest.raises(OverlapError):
        store(
            **make_store_kwargs(
                value="Go",
                valid_from=datetime(2026, 6, 1, tzinfo=timezone.utc),
                valid_to=None,
                source_id="conversation_002",
            ),
            db_path=database_path,
        )


def test_different_memory_keys_do_not_conflict(database_path):
    backend = store(
        **make_store_kwargs(
            valid_to=None,
        ),
        db_path=database_path,
    )

    db_memory = store(
        **make_store_kwargs(
            memory_key="primary_database",
            value="PostgreSQL",
            valid_to=None,
            source_id="conversation_002",
        ),
        db_path=database_path,
    )

    backend_timeline = list_by_key(
        "primary_backend_language",
        database_path,
    )

    database_timeline = list_by_key(
        "primary_database",
        database_path,
    )

    assert backend.memory_id != db_memory.memory_id
    assert len(backend_timeline) == 1
    assert len(database_timeline) == 1
    assert backend_timeline[0].value == "Node.js"
    assert database_timeline[0].value == "PostgreSQL"


def test_store_generates_unique_ids(database_path):
    first = store(
        **make_store_kwargs(
            valid_to=datetime(2026, 6, 1, tzinfo=timezone.utc),
        ),
        db_path=database_path,
    )

    second = store(
        **make_store_kwargs(
            value="Go",
            valid_from=datetime(2026, 6, 1, tzinfo=timezone.utc),
            valid_to=None,
            source_id="conversation_002",
            supersedes_id=first.memory_id,
        ),
        db_path=database_path,
    )

    assert first.memory_id != second.memory_id


def test_rejected_store_leaves_database_unchanged(database_path):
    store(
        **make_store_kwargs(
            valid_to=None,
        ),
        db_path=database_path,
    )

    before = list_by_key(
        "primary_backend_language",
        database_path,
    )

    with pytest.raises(OverlapError):
        store(
            **make_store_kwargs(
                value="Python",
                valid_from=datetime(2026, 3, 1, tzinfo=timezone.utc),
                valid_to=None,
                source_id="conversation_002",
            ),
            db_path=database_path,
        )

    after = list_by_key(
        "primary_backend_language",
        database_path,
    )

    assert len(after) == len(before)
    assert after == before