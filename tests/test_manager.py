import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from app.memory.manager import (
    NoOpenMemoryError,
    OverlapError,
    _overlaps,
    get_at_time,
    store,
    supersede,
)
from app.memory.models import (
    EvidenceType,
    MemoryStatus,
    MemoryType,
    SourceType,
    TimePrecision,
)
from app.storage.database import initialize_database
from app.storage.memory_repository import get, list_by_key


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
        "valid_from": datetime(
            2026,
            1,
            1,
            tzinfo=timezone.utc,
        ),
        "valid_to": datetime(
            2026,
            6,
            1,
            tzinfo=timezone.utc,
        ),
        "precision": TimePrecision.DAY,
        "source_type": SourceType.CONVERSATION,
        "source_id": "conversation_001",
        "evidence_type": EvidenceType.EXPLICIT,
        "confidence": 0.98,
    }

    data.update(overrides)
    return data


def make_supersede_kwargs(**overrides):
    data = {
        "memory_key": "primary_backend_language",
        "subject": "user",
        "attribute": "uses",
        "value": "Go",
        "memory_type": MemoryType.SKILL,
        "valid_from": datetime(
            2026,
            6,
            1,
            tzinfo=timezone.utc,
        ),
        "precision": TimePrecision.DAY,
        "source_type": SourceType.CONVERSATION,
        "source_id": "conversation_002",
        "evidence_type": EvidenceType.EXPLICIT,
        "confidence": 0.99,
    }

    data.update(overrides)
    return data


def test_adjacent_intervals_do_not_overlap():
    june = datetime(
        2026,
        6,
        1,
        tzinfo=timezone.utc,
    )

    assert _overlaps(
        datetime(
            2026,
            1,
            1,
            tzinfo=timezone.utc,
        ),
        june,
        june,
        None,
    ) is False


def test_overlapping_intervals_overlap():
    assert _overlaps(
        datetime(
            2026,
            1,
            1,
            tzinfo=timezone.utc,
        ),
        datetime(
            2026,
            6,
            1,
            tzinfo=timezone.utc,
        ),
        datetime(
            2026,
            5,
            1,
            tzinfo=timezone.utc,
        ),
        None,
    ) is True


def test_two_open_ended_intervals_overlap():
    assert _overlaps(
        datetime(
            2026,
            1,
            1,
            tzinfo=timezone.utc,
        ),
        None,
        datetime(
            2026,
            6,
            1,
            tzinfo=timezone.utc,
        ),
        None,
    ) is True


def test_non_overlapping_historical_intervals():
    assert _overlaps(
        datetime(
            2026,
            1,
            1,
            tzinfo=timezone.utc,
        ),
        datetime(
            2026,
            3,
            1,
            tzinfo=timezone.utc,
        ),
        datetime(
            2026,
            3,
            1,
            tzinfo=timezone.utc,
        ),
        datetime(
            2026,
            6,
            1,
            tzinfo=timezone.utc,
        ),
    ) is False


def test_inner_interval_overlaps_outer_interval():
    assert _overlaps(
        datetime(
            2026,
            1,
            1,
            tzinfo=timezone.utc,
        ),
        datetime(
            2026,
            12,
            1,
            tzinfo=timezone.utc,
        ),
        datetime(
            2026,
            3,
            1,
            tzinfo=timezone.utc,
        ),
        datetime(
            2026,
            6,
            1,
            tzinfo=timezone.utc,
        ),
    ) is True


def test_one_microsecond_overlap():
    june = datetime(
        2026,
        6,
        1,
        tzinfo=timezone.utc,
    )

    assert _overlaps(
        datetime(
            2026,
            1,
            1,
            tzinfo=timezone.utc,
        ),
        june,
        june - timedelta(microseconds=1),
        None,
    ) is True


def test_store_node_js_then_go(database_path):
    node = store(
        **make_store_kwargs(
            value="Node.js",
            valid_from=datetime(
                2026,
                1,
                1,
                tzinfo=timezone.utc,
            ),
            valid_to=datetime(
                2026,
                6,
                1,
                tzinfo=timezone.utc,
            ),
            source_id="conversation_001",
        ),
        db_path=database_path,
    )

    go = store(
        **make_store_kwargs(
            value="Go",
            valid_from=datetime(
                2026,
                6,
                1,
                tzinfo=timezone.utc,
            ),
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
        datetime(
            2026,
            3,
            1,
            tzinfo=timezone.utc,
        ),
        database_path,
    )

    july = get_at_time(
        "primary_backend_language",
        datetime(
            2026,
            7,
            1,
            tzinfo=timezone.utc,
        ),
        database_path,
    )

    assert march is not None
    assert march.value == "Node.js"

    assert july is not None
    assert july.value == "Go"


def test_closed_interval_gets_historical_status(database_path):
    memory = store(
        **make_store_kwargs(
            valid_to=datetime(
                2026,
                6,
                1,
                tzinfo=timezone.utc,
            ),
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
            valid_to=datetime(
                2026,
                6,
                1,
                tzinfo=timezone.utc,
            ),
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
                valid_from=datetime(
                    2026,
                    5,
                    1,
                    tzinfo=timezone.utc,
                ),
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
                valid_from=datetime(
                    2026,
                    6,
                    1,
                    tzinfo=timezone.utc,
                ),
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
            valid_to=datetime(
                2026,
                6,
                1,
                tzinfo=timezone.utc,
            ),
        ),
        db_path=database_path,
    )

    second = store(
        **make_store_kwargs(
            value="Go",
            valid_from=datetime(
                2026,
                6,
                1,
                tzinfo=timezone.utc,
            ),
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
                valid_from=datetime(
                    2026,
                    3,
                    1,
                    tzinfo=timezone.utc,
                ),
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


def test_supersede_closes_old_memory_and_creates_new_memory(
    database_path,
):
    old = store(
        **make_store_kwargs(
            value="Node.js",
            valid_from=datetime(
                2026,
                1,
                1,
                tzinfo=timezone.utc,
            ),
            valid_to=None,
            source_id="conversation_001",
        ),
        db_path=database_path,
    )

    new = supersede(
        **make_supersede_kwargs(),
        db_path=database_path,
    )

    timeline = list_by_key(
        "primary_backend_language",
        database_path,
    )

    assert len(timeline) == 2

    assert timeline[0].memory_id == old.memory_id
    assert timeline[0].value == "Node.js"
    assert timeline[0].valid_to == new.valid_from
    assert timeline[0].status == MemoryStatus.SUPERSEDED

    assert timeline[1].memory_id == new.memory_id
    assert timeline[1].value == "Go"
    assert timeline[1].valid_from == datetime(
        2026,
        6,
        1,
        tzinfo=timezone.utc,
    )
    assert timeline[1].valid_to is None
    assert timeline[1].status == MemoryStatus.ACTIVE
    assert timeline[1].supersedes_id == old.memory_id


def test_supersede_temporal_queries_return_correct_memory(
    database_path,
):
    store(
        **make_store_kwargs(
            value="Node.js",
            valid_from=datetime(
                2026,
                1,
                1,
                tzinfo=timezone.utc,
            ),
            valid_to=None,
            source_id="conversation_001",
        ),
        db_path=database_path,
    )

    supersede(
        **make_supersede_kwargs(),
        db_path=database_path,
    )

    march = get_at_time(
        "primary_backend_language",
        datetime(
            2026,
            3,
            1,
            tzinfo=timezone.utc,
        ),
        database_path,
    )

    boundary = get_at_time(
        "primary_backend_language",
        datetime(
            2026,
            6,
            1,
            tzinfo=timezone.utc,
        ),
        database_path,
    )

    july = get_at_time(
        "primary_backend_language",
        datetime(
            2026,
            7,
            1,
            tzinfo=timezone.utc,
        ),
        database_path,
    )

    assert march is not None
    assert march.value == "Node.js"

    assert boundary is not None
    assert boundary.value == "Go"

    assert july is not None
    assert july.value == "Go"


def test_supersede_requires_open_memory_on_empty_key(database_path):
    with pytest.raises(NoOpenMemoryError):
        supersede(
            **make_supersede_kwargs(),
            db_path=database_path,
        )


def test_supersede_requires_open_memory_when_only_memory_is_closed(
    database_path,
):
    store(
        **make_store_kwargs(
            value="Node.js",
            valid_from=datetime(
                2026,
                1,
                1,
                tzinfo=timezone.utc,
            ),
            valid_to=datetime(
                2026,
                6,
                1,
                tzinfo=timezone.utc,
            ),
        ),
        db_path=database_path,
    )

    with pytest.raises(NoOpenMemoryError):
        supersede(
            **make_supersede_kwargs(
                valid_from=datetime(
                    2026,
                    7,
                    1,
                    tzinfo=timezone.utc,
                ),
            ),
            db_path=database_path,
        )


def test_supersede_rejects_earlier_valid_from(database_path):
    store(
        **make_store_kwargs(
            value="Node.js",
            valid_from=datetime(
                2026,
                6,
                1,
                tzinfo=timezone.utc,
            ),
            valid_to=None,
        ),
        db_path=database_path,
    )

    before = list_by_key(
        "primary_backend_language",
        database_path,
    )

    with pytest.raises(ValueError):
        supersede(
            **make_supersede_kwargs(
                valid_from=datetime(
                    2026,
                    5,
                    1,
                    tzinfo=timezone.utc,
                ),
            ),
            db_path=database_path,
        )

    after = list_by_key(
        "primary_backend_language",
        database_path,
    )

    assert after == before


def test_supersede_rejects_equal_valid_from(database_path):
    valid_from = datetime(
        2026,
        6,
        1,
        tzinfo=timezone.utc,
    )

    store(
        **make_store_kwargs(
            value="Node.js",
            valid_from=valid_from,
            valid_to=None,
        ),
        db_path=database_path,
    )

    before = list_by_key(
        "primary_backend_language",
        database_path,
    )

    with pytest.raises(ValueError):
        supersede(
            **make_supersede_kwargs(
                valid_from=valid_from,
            ),
            db_path=database_path,
        )

    after = list_by_key(
        "primary_backend_language",
        database_path,
    )

    assert after == before


def test_supersede_rejects_naive_datetime(database_path):
    with pytest.raises(ValueError, match="timezone-aware"):
        supersede(
            **make_supersede_kwargs(
                valid_from=datetime(
                    2026,
                    6,
                    1,
                ),
            ),
            db_path=database_path,
        )


def test_failed_supersede_leaves_old_memory_open(
    database_path,
    monkeypatch,
):
    old = store(
        **make_store_kwargs(
            value="Node.js",
            valid_from=datetime(
                2026,
                1,
                1,
                tzinfo=timezone.utc,
            ),
            valid_to=None,
            source_id="conversation_001",
        ),
        db_path=database_path,
    )

    # Create a DIFFERENT existing memory.
    # Its ID will be used to force the INSERT collision.
    conflicting = store(
        **make_store_kwargs(
            memory_key="other_memory_key",
            value="Existing",
            valid_from=datetime(
                2026,
                1,
                1,
                tzinfo=timezone.utc,
            ),
            valid_to=None,
            source_id="conversation_conflict",
        ),
        db_path=database_path,
    )

    # Force supersede() to generate the already-existing
    # conflicting memory ID.
    monkeypatch.setattr(
        "app.memory.manager.uuid4",
        lambda: type(
            "U",
            (),
            {
                "hex": conflicting.memory_id,
            },
        )(),
    )

    with pytest.raises(sqlite3.IntegrityError):
        supersede(
            **make_supersede_kwargs(
                value="Go",
                valid_from=datetime(
                    2026,
                    6,
                    1,
                    tzinfo=timezone.utc,
                ),
            ),
            db_path=database_path,
        )

    # The UPDATE to the old memory must have been rolled back.
    after = get(
        old.memory_id,
        database_path,
    )

    assert after is not None
    assert after.valid_to is None
    assert after.status == MemoryStatus.ACTIVE

    # No new memory should have been committed.
    timeline = list_by_key(
        "primary_backend_language",
        database_path,
    )

    assert len(timeline) == 1
    assert timeline[0].memory_id == old.memory_id

    # The unrelated conflicting memory must still exist.
    conflicting_after = get(
        conflicting.memory_id,
        database_path,
    )

    assert conflicting_after is not None
    assert conflicting_after.value == "Existing"