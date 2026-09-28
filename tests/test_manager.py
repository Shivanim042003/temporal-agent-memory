import sqlite3
from datetime import datetime, timedelta, timezone
from uuid import uuid4
import pytest

from app.storage.database import initialize_database

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
from app.storage.memory_repository import get, list_by_key


def make_store_kwargs(
    *,
    memory_key="primary_backend_language",
    subject="user",
    attribute="programming_language",
    value="Node.js",
    memory_type=MemoryType.SKILL,
    valid_from=datetime(
        2026,
        1,
        1,
        tzinfo=timezone.utc,
    ),
    valid_to=None,
    precision=TimePrecision.EXACT,
    source_type=SourceType.CONVERSATION,
    source_id="conversation_001",
    evidence_type=EvidenceType.EXPLICIT,
    confidence=1.0,
    supersedes_id=None,
):
    return {
        "memory_key": memory_key,
        "subject": subject,
        "attribute": attribute,
        "value": value,
        "memory_type": memory_type,
        "valid_from": valid_from,
        "valid_to": valid_to,
        "precision": precision,
        "source_type": source_type,
        "source_id": source_id,
        "evidence_type": evidence_type,
        "confidence": confidence,
        "supersedes_id": supersedes_id,
    }


def make_supersede_kwargs(
    *,
    memory_key="primary_backend_language",
    subject="user",
    attribute="programming_language",
    value="Go",
    memory_type=MemoryType.SKILL,
    valid_from=datetime(
        2026,
        6,
        1,
        tzinfo=timezone.utc,
    ),
    precision=TimePrecision.EXACT,
    source_type=SourceType.CONVERSATION,
    source_id="conversation_002",
    evidence_type=EvidenceType.EXPLICIT,
    confidence=1.0,
):
    return {
        "memory_key": memory_key,
        "subject": subject,
        "attribute": attribute,
        "value": value,
        "memory_type": memory_type,
        "valid_from": valid_from,
        "precision": precision,
        "source_type": source_type,
        "source_id": source_id,
        "evidence_type": evidence_type,
        "confidence": confidence,
    }


@pytest.fixture
def database_path(tmp_path):
    path = tmp_path / "memory.db"
    initialize_database(path)
    return path


def test_overlaps_open_intervals():
    assert _overlaps(
        datetime(2026, 1, 1, tzinfo=timezone.utc),
        None,
        datetime(2026, 2, 1, tzinfo=timezone.utc),
        None,
    )


def test_overlaps_closed_intervals():
    assert _overlaps(
        datetime(2026, 1, 1, tzinfo=timezone.utc),
        datetime(2026, 6, 1, tzinfo=timezone.utc),
        datetime(2026, 5, 1, tzinfo=timezone.utc),
        datetime(2026, 7, 1, tzinfo=timezone.utc),
    )


def test_adjacent_intervals_do_not_overlap():
    assert not _overlaps(
        datetime(2026, 1, 1, tzinfo=timezone.utc),
        datetime(2026, 6, 1, tzinfo=timezone.utc),
        datetime(2026, 6, 1, tzinfo=timezone.utc),
        None,
    )


def test_interval_before_another_does_not_overlap():
    assert not _overlaps(
        datetime(2026, 1, 1, tzinfo=timezone.utc),
        datetime(2026, 2, 1, tzinfo=timezone.utc),
        datetime(2026, 3, 1, tzinfo=timezone.utc),
        datetime(2026, 4, 1, tzinfo=timezone.utc),
    )


def test_interval_after_another_does_not_overlap():
    assert not _overlaps(
        datetime(2026, 3, 1, tzinfo=timezone.utc),
        datetime(2026, 4, 1, tzinfo=timezone.utc),
        datetime(2026, 1, 1, tzinfo=timezone.utc),
        datetime(2026, 2, 1, tzinfo=timezone.utc),
    )


def test_store_creates_active_memory_when_valid_to_is_none(
    database_path,
):
    memory = store(
        **make_store_kwargs(),
        db_path=database_path,
    )

    assert memory.memory_id
    assert memory.memory_key == "primary_backend_language"
    assert memory.value == "Node.js"
    assert memory.status == MemoryStatus.ACTIVE
    assert memory.valid_to is None


def test_store_creates_historical_memory_when_valid_to_exists(
    database_path,
):
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
    assert memory.valid_to == datetime(
        2026,
        6,
        1,
        tzinfo=timezone.utc,
    )


def test_store_persists_memory(
    database_path,
):
    memory = store(
        **make_store_kwargs(),
        db_path=database_path,
    )

    retrieved = get(
        memory.memory_id,
        database_path,
    )

    assert retrieved == memory


def test_store_allows_adjacent_interval(
    database_path,
):
    store(
        **make_store_kwargs(
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
        ),
        db_path=database_path,
    )

    assert second.value == "Go"


def test_store_rejects_overlapping_interval(
    database_path,
):
    store(
        **make_store_kwargs(),
        db_path=database_path,
    )

    with pytest.raises(OverlapError):
        store(
            **make_store_kwargs(
                value="Go",
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


def test_store_rejects_interval_containing_existing_memory(
    database_path,
):
    store(
        **make_store_kwargs(
            valid_from=datetime(
                2026,
                3,
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

    with pytest.raises(OverlapError):
        store(
            **make_store_kwargs(
                value="Go",
                valid_from=datetime(
                    2026,
                    1,
                    1,
                    tzinfo=timezone.utc,
                ),
                valid_to=datetime(
                    2026,
                    8,
                    1,
                    tzinfo=timezone.utc,
                ),
                source_id="conversation_002",
            ),
            db_path=database_path,
        )


def test_store_rejects_interval_inside_existing_memory(
    database_path,
):
    store(
        **make_store_kwargs(
            valid_from=datetime(
                2026,
                1,
                1,
                tzinfo=timezone.utc,
            ),
            valid_to=datetime(
                2026,
                12,
                1,
                tzinfo=timezone.utc,
            ),
        ),
        db_path=database_path,
    )

    with pytest.raises(OverlapError):
        store(
            **make_store_kwargs(
                value="Go",
                valid_from=datetime(
                    2026,
                    3,
                    1,
                    tzinfo=timezone.utc,
                ),
                valid_to=datetime(
                    2026,
                    6,
                    1,
                    tzinfo=timezone.utc,
                ),
                source_id="conversation_002",
            ),
            db_path=database_path,
        )


def test_store_rejects_open_interval_overlapping_existing_memory(
    database_path,
):
    store(
        **make_store_kwargs(
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

    with pytest.raises(OverlapError):
        store(
            **make_store_kwargs(
                value="Go",
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


def test_store_rejects_closed_interval_overlapping_open_memory(
    database_path,
):
    store(
        **make_store_kwargs(),
        db_path=database_path,
    )

    with pytest.raises(OverlapError):
        store(
            **make_store_kwargs(
                value="Go",
                valid_from=datetime(
                    2026,
                    5,
                    1,
                    tzinfo=timezone.utc,
                ),
                valid_to=datetime(
                    2026,
                    7,
                    1,
                    tzinfo=timezone.utc,
                ),
                source_id="conversation_002",
            ),
            db_path=database_path,
        )


def test_store_rejected_overlap_leaves_existing_memory_unchanged(
    database_path,
):
    original = store(
        **make_store_kwargs(),
        db_path=database_path,
    )

    with pytest.raises(OverlapError):
        store(
            **make_store_kwargs(
                value="Go",
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

    retrieved = get(
        original.memory_id,
        database_path,
    )

    assert retrieved == original


def test_store_with_different_memory_key_does_not_overlap(
    database_path,
):
    first = store(
        **make_store_kwargs(
            memory_key="primary_backend_language",
        ),
        db_path=database_path,
    )

    second = store(
        **make_store_kwargs(
            memory_key="secondary_backend_language",
            value="Go",
            source_id="conversation_002",
        ),
        db_path=database_path,
    )

    assert first.value == "Node.js"
    assert second.value == "Go"


def test_store_with_timezone_aware_datetime_normalizes_to_utc(
    database_path,
):
    memory = store(
        **make_store_kwargs(
            valid_from=datetime(
                2026,
                1,
                1,
                5,
                0,
                tzinfo=timezone.utc,
            ),
        ),
        db_path=database_path,
    )

    assert memory.valid_from.tzinfo == timezone.utc


def test_store_preserves_confidence(
    database_path,
):
    memory = store(
        **make_store_kwargs(
            confidence=0.75,
        ),
        db_path=database_path,
    )

    assert memory.confidence == 0.75


def test_store_preserves_provenance(
    database_path,
):
    memory = store(
        **make_store_kwargs(
            source_id="conversation_123",
            evidence_type=EvidenceType.INFERRED,
        ),
        db_path=database_path,
    )

    assert memory.source_id == "conversation_123"
    assert memory.evidence_type == EvidenceType.INFERRED


def test_store_rejected_duplicate_id_leaves_database_unchanged(
    database_path,
    monkeypatch,
):
    original = store(
        **make_store_kwargs(),
        db_path=database_path,
    )

    original_uuid4 = uuid4

    class FixedUUID:
        hex = original.memory_id

    monkeypatch.setattr(
        "app.memory.manager.uuid4",
        lambda: FixedUUID(),
    )

    with pytest.raises(sqlite3.IntegrityError):
        store(
            **make_store_kwargs(
                memory_key="different_memory_key",
                value="Go",
                source_id="conversation_002",
            ),
            db_path=database_path,
        )

    retrieved = get(
        original.memory_id,
        database_path,
    )

    assert retrieved == original

    monkeypatch.setattr(
        "app.memory.manager.uuid4",
        original_uuid4,
    )


def test_supersede_closes_current_memory(
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

    retrieved_old = get(
        old.memory_id,
        database_path,
    )

    assert retrieved_old is not None
    assert retrieved_old.valid_to == new.valid_from
    assert retrieved_old.status == MemoryStatus.SUPERSEDED


def test_supersede_creates_new_active_memory(
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

    new = supersede(
        **make_supersede_kwargs(),
        db_path=database_path,
    )

    assert new.value == "Go"
    assert new.valid_from == datetime(
        2026,
        6,
        1,
        tzinfo=timezone.utc,
    )
    assert new.valid_to is None
    assert new.status == MemoryStatus.ACTIVE


def test_supersede_links_new_memory_to_old_memory(
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

    assert new.supersedes_id == old.memory_id


def test_supersede_requires_open_memory(
    database_path,
):
    with pytest.raises(NoOpenMemoryError):
        supersede(
            **make_supersede_kwargs(),
            db_path=database_path,
        )


def test_supersede_rejects_non_increasing_valid_from(
    database_path,
):
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
            source_id="conversation_001",
        ),
        db_path=database_path,
    )

    with pytest.raises(ValueError):
        supersede(
            **make_supersede_kwargs(
                valid_from=datetime(
                    2026,
                    6,
                    1,
                    tzinfo=timezone.utc,
                ),
            ),
            db_path=database_path,
        )


def test_supersede_rejects_earlier_valid_from(
    database_path,
):
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
            source_id="conversation_001",
        ),
        db_path=database_path,
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


def test_supersede_rejects_when_multiple_open_memories_exist(
    database_path,
):
    store(
        **make_store_kwargs(
            memory_key="primary_backend_language",
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

    connection = sqlite3.connect(database_path)

    try:
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
                "manual-open-memory",
                "primary_backend_language",
                "user",
                "programming_language",
                "Go",
                "skill",
                "2026-02-01T00:00:00+00:00",
                None,
                "exact",
                "2026-02-01T00:00:00+00:00",
                "conversation",
                "conversation_002",
                "explicit",
                1.0,
                "active",
                None,
            ),
        )

        connection.commit()

    finally:
        connection.close()

    with pytest.raises(OverlapError):
        supersede(
            **make_supersede_kwargs(),
            db_path=database_path,
        )


def test_supersede_preserves_timeline(
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

    conflicting = store(
        **make_store_kwargs(
            memory_key="conflicting_memory",
            value="Python",
            valid_from=datetime(
                2026,
                1,
                1,
                tzinfo=timezone.utc,
            ),
            valid_to=None,
            source_id="conversation_003",
        ),
        db_path=database_path,
    )

    class FixedUUID:
        hex = conflicting.memory_id

    monkeypatch.setattr(
        "app.memory.manager.uuid4",
        lambda: FixedUUID(),
    )

    with pytest.raises(sqlite3.IntegrityError):
        supersede(
            **make_supersede_kwargs(),
            db_path=database_path,
        )

    retrieved_old = get(
        old.memory_id,
        database_path,
    )

    assert retrieved_old is not None
    assert retrieved_old.valid_to is None
    assert retrieved_old.status == MemoryStatus.ACTIVE

    retrieved_conflicting = get(
        conflicting.memory_id,
        database_path,
    )

    assert retrieved_conflicting is not None
    assert retrieved_conflicting.valid_to is None
    assert retrieved_conflicting.status == MemoryStatus.ACTIVE