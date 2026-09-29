from datetime import datetime, timezone

import pytest

from app.memory.manager import (
    InvalidConsolidationError,
    MemoryNotFoundError,
    consolidate,
)
from app.memory.models import (
    EvidenceType,
    MemoryStatus,
    MemoryType,
    SourceType,
    TimePrecision,
)
from app.storage.database import initialize_database
from app.storage.memory_repository import get, insert


@pytest.fixture
def database_path(tmp_path):
    path = tmp_path / "memory.db"
    initialize_database(path)
    return path


def make_memory(
    *,
    memory_id: str,
    recorded_at: datetime,
    value: str = "Python",
    valid_from: datetime | None = None,
    valid_to: datetime | None = None,
    status: MemoryStatus = MemoryStatus.ACTIVE,
    canonical_memory_id: str | None = None,
):
    if valid_from is None:
        valid_from = datetime(
            2026,
            1,
            1,
            tzinfo=timezone.utc,
        )

    return {
        "memory_id": memory_id,
        "memory_key": "user:language",
        "subject": "user",
        "attribute": "programming_language",
        "value": value,
        "memory_type": MemoryType.SKILL,
        "valid_from": valid_from,
        "valid_to": valid_to,
        "precision": TimePrecision.EXACT,
        "recorded_at": recorded_at,
        "source_type": SourceType.CONVERSATION,
        "source_id": f"conversation-{memory_id}",
        "evidence_type": EvidenceType.EXPLICIT,
        "confidence": 0.95,
        "status": status,
        "supersedes_id": None,
        "canonical_memory_id": canonical_memory_id,
    }


def insert_memory(database_path, **kwargs):
    from app.memory.models import Memory

    memory = Memory(**make_memory(**kwargs))
    insert(memory, database_path)
    return memory


def test_consolidate_marks_redundant_memories(
    database_path,
):
    first = insert_memory(
        database_path,
        memory_id="mem_a",
        recorded_at=datetime(
            2026,
            1,
            1,
            tzinfo=timezone.utc,
        ),
    )

    second = insert_memory(
        database_path,
        memory_id="mem_b",
        recorded_at=datetime(
            2026,
            1,
            2,
            tzinfo=timezone.utc,
        ),
    )

    third = insert_memory(
        database_path,
        memory_id="mem_c",
        recorded_at=datetime(
            2026,
            1,
            3,
            tzinfo=timezone.utc,
        ),
    )

    canonical = consolidate(
        [first.memory_id, second.memory_id, third.memory_id],
        db_path=database_path,
    )

    assert canonical.memory_id == "mem_a"

    stored_first = get(
        "mem_a",
        database_path,
    )
    stored_second = get(
        "mem_b",
        database_path,
    )
    stored_third = get(
        "mem_c",
        database_path,
    )

    assert stored_first is not None
    assert stored_second is not None
    assert stored_third is not None

    assert stored_first.status == MemoryStatus.ACTIVE
    assert stored_first.canonical_memory_id is None

    assert stored_second.status == MemoryStatus.CONSOLIDATED
    assert stored_second.canonical_memory_id == "mem_a"

    assert stored_third.status == MemoryStatus.CONSOLIDATED
    assert stored_third.canonical_memory_id == "mem_a"


def test_consolidate_supports_explicit_canonical_memory(
    database_path,
):
    first = insert_memory(
        database_path,
        memory_id="mem_a",
        recorded_at=datetime(
            2026,
            1,
            1,
            tzinfo=timezone.utc,
        ),
    )

    second = insert_memory(
        database_path,
        memory_id="mem_b",
        recorded_at=datetime(
            2026,
            1,
            2,
            tzinfo=timezone.utc,
        ),
    )

    third = insert_memory(
        database_path,
        memory_id="mem_c",
        recorded_at=datetime(
            2026,
            1,
            3,
            tzinfo=timezone.utc,
        ),
    )

    canonical = consolidate(
        [first.memory_id, second.memory_id, third.memory_id],
        canonical_memory_id="mem_b",
        db_path=database_path,
    )

    assert canonical.memory_id == "mem_b"

    stored_first = get("mem_a", database_path)
    stored_second = get("mem_b", database_path)
    stored_third = get("mem_c", database_path)

    assert stored_first is not None
    assert stored_second is not None
    assert stored_third is not None

    assert stored_first.status == MemoryStatus.CONSOLIDATED
    assert stored_first.canonical_memory_id == "mem_b"

    assert stored_second.status == MemoryStatus.ACTIVE
    assert stored_second.canonical_memory_id is None

    assert stored_third.status == MemoryStatus.CONSOLIDATED
    assert stored_third.canonical_memory_id == "mem_b"


def test_consolidate_rejects_unknown_memory(
    database_path,
):
    memory = insert_memory(
        database_path,
        memory_id="mem_a",
        recorded_at=datetime(
            2026,
            1,
            1,
            tzinfo=timezone.utc,
        ),
    )

    with pytest.raises(MemoryNotFoundError):
        consolidate(
            [memory.memory_id, "does_not_exist"],
            db_path=database_path,
        )


def test_consolidate_rejects_fewer_than_two_memories(
    database_path,
):
    memory = insert_memory(
        database_path,
        memory_id="mem_a",
        recorded_at=datetime(
            2026,
            1,
            1,
            tzinfo=timezone.utc,
        ),
    )

    with pytest.raises(InvalidConsolidationError):
        consolidate(
            [memory.memory_id],
            db_path=database_path,
        )


def test_consolidate_rejects_duplicate_memory_ids(
    database_path,
):
    memory_a = insert_memory(
        database_path,
        memory_id="mem_a",
        recorded_at=datetime(
            2026,
            1,
            1,
            tzinfo=timezone.utc,
        ),
    )

    memory_b = insert_memory(
        database_path,
        memory_id="mem_b",
        recorded_at=datetime(
            2026,
            1,
            2,
            tzinfo=timezone.utc,
        ),
    )

    with pytest.raises(InvalidConsolidationError):
        consolidate(
            [memory_a.memory_id, memory_a.memory_id, memory_b.memory_id],
            db_path=database_path,
        )


def test_consolidate_rejects_different_values(
    database_path,
):
    insert_memory(
        database_path,
        memory_id="mem_a",
        recorded_at=datetime(
            2026,
            1,
            1,
            tzinfo=timezone.utc,
        ),
        value="Python",
    )

    insert_memory(
        database_path,
        memory_id="mem_b",
        recorded_at=datetime(
            2026,
            1,
            2,
            tzinfo=timezone.utc,
        ),
        value="Java",
    )

    with pytest.raises(InvalidConsolidationError):
        consolidate(
            ["mem_a", "mem_b"],
            db_path=database_path,
        )


def test_consolidate_rejects_different_valid_from(
    database_path,
):
    insert_memory(
        database_path,
        memory_id="mem_a",
        recorded_at=datetime(
            2026,
            1,
            1,
            tzinfo=timezone.utc,
        ),
        valid_from=datetime(
            2026,
            1,
            1,
            tzinfo=timezone.utc,
        ),
    )

    insert_memory(
        database_path,
        memory_id="mem_b",
        recorded_at=datetime(
            2026,
            1,
            2,
            tzinfo=timezone.utc,
        ),
        valid_from=datetime(
            2026,
            2,
            1,
            tzinfo=timezone.utc,
        ),
    )

    with pytest.raises(InvalidConsolidationError):
        consolidate(
            ["mem_a", "mem_b"],
            db_path=database_path,
        )


def test_consolidate_rejects_already_consolidated_memory(
    database_path,
):
    insert_memory(
        database_path,
        memory_id="mem_a",
        recorded_at=datetime(
            2026,
            1,
            1,
            tzinfo=timezone.utc,
        ),
    )

    insert_memory(
        database_path,
        memory_id="mem_b",
        recorded_at=datetime(
            2026,
            1,
            2,
            tzinfo=timezone.utc,
        ),
        status=MemoryStatus.CONSOLIDATED,
        canonical_memory_id="mem_a",
    )

    with pytest.raises(InvalidConsolidationError):
        consolidate(
            ["mem_a", "mem_b"],
            db_path=database_path,
        )


def test_consolidate_rejects_canonical_memory_outside_candidates(
    database_path,
):
    insert_memory(
        database_path,
        memory_id="mem_a",
        recorded_at=datetime(
            2026,
            1,
            1,
            tzinfo=timezone.utc,
        ),
    )

    insert_memory(
        database_path,
        memory_id="mem_b",
        recorded_at=datetime(
            2026,
            1,
            2,
            tzinfo=timezone.utc,
        ),
    )

    insert_memory(
        database_path,
        memory_id="mem_c",
        recorded_at=datetime(
            2026,
            1,
            3,
            tzinfo=timezone.utc,
        ),
    )

    with pytest.raises(InvalidConsolidationError):
        consolidate(
            ["mem_a", "mem_b"],
            canonical_memory_id="mem_c",
            db_path=database_path,
        )


def test_consolidate_failure_leaves_database_unchanged(
    database_path,
):
    insert_memory(
        database_path,
        memory_id="mem_a",
        recorded_at=datetime(
            2026,
            1,
            1,
            tzinfo=timezone.utc,
        ),
    )

    insert_memory(
        database_path,
        memory_id="mem_b",
        recorded_at=datetime(
            2026,
            1,
            2,
            tzinfo=timezone.utc,
        ),
        value="Java",
    )

    with pytest.raises(InvalidConsolidationError):
        consolidate(
            ["mem_a", "mem_b"],
            db_path=database_path,
        )

    stored_a = get("mem_a", database_path)
    stored_b = get("mem_b", database_path)

    assert stored_a is not None
    assert stored_b is not None

    assert stored_a.status == MemoryStatus.ACTIVE
    assert stored_a.canonical_memory_id is None

    assert stored_b.status == MemoryStatus.ACTIVE
    assert stored_b.canonical_memory_id is None