from datetime import datetime, timezone

import pytest

from app.memory.manager import (
    InvalidConsolidationError,
    MemoryNotFoundError,
    consolidate,
    discard,
    get_at_time,
    get_current,
    get_range,
    hybrid_search,
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
        [
            first.memory_id,
            second.memory_id,
            third.memory_id,
        ],
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
        [
            first.memory_id,
            second.memory_id,
            third.memory_id,
        ],
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
            [
                memory.memory_id,
                "does_not_exist",
            ],
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
            [
                memory_a.memory_id,
                memory_a.memory_id,
                memory_b.memory_id,
            ],
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
            [
                "mem_a",
                "mem_b",
            ],
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
            [
                "mem_a",
                "mem_b",
            ],
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
            [
                "mem_a",
                "mem_b",
            ],
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
            [
                "mem_a",
                "mem_b",
            ],
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
            [
                "mem_a",
                "mem_b",
            ],
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


def test_consolidate_preserves_redundant_memories_as_evidence(
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
        [
            first.memory_id,
            second.memory_id,
            third.memory_id,
        ],
        db_path=database_path,
    )

    from app.storage.memory_repository import list_consolidated_into

    evidence = list_consolidated_into(
        canonical.memory_id,
        database_path,
    )

    assert [
        memory.memory_id
        for memory in evidence
    ] == [
        "mem_b",
        "mem_c",
    ]

    assert all(
        memory.status == MemoryStatus.CONSOLIDATED
        for memory in evidence
    )

    assert all(
        memory.canonical_memory_id == canonical.memory_id
        for memory in evidence
    )


def test_get_current_excludes_consolidated_memories(
    database_path,
):
    canonical = insert_memory(
        database_path,
        memory_id="mem_canonical",
        recorded_at=datetime(
            2026,
            1,
            1,
            tzinfo=timezone.utc,
        ),
    )

    consolidated = insert_memory(
        database_path,
        memory_id="mem_consolidated",
        recorded_at=datetime(
            2026,
            1,
            2,
            tzinfo=timezone.utc,
        ),
        status=MemoryStatus.CONSOLIDATED,
        canonical_memory_id=canonical.memory_id,
    )

    result = get_current(
        canonical.memory_key,
        database_path,
    )

    assert result is not None
    assert result.memory_id == canonical.memory_id
    assert result.memory_id != consolidated.memory_id


def test_get_range_excludes_consolidated_memories(
    database_path,
):
    canonical = insert_memory(
        database_path,
        memory_id="mem_canonical",
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
        valid_to=datetime(
            2026,
            6,
            1,
            tzinfo=timezone.utc,
        ),
        status=MemoryStatus.HISTORICAL,
    )

    consolidated = insert_memory(
        database_path,
        memory_id="mem_consolidated",
        recorded_at=datetime(
            2026,
            1,
            2,
            tzinfo=timezone.utc,
        ),
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
        status=MemoryStatus.CONSOLIDATED,
        canonical_memory_id=canonical.memory_id,
    )

    result = get_range(
        canonical.memory_key,
        datetime(
            2026,
            2,
            1,
            tzinfo=timezone.utc,
        ),
        datetime(
            2026,
            3,
            1,
            tzinfo=timezone.utc,
        ),
        database_path,
    )

    assert [
        memory.memory_id
        for memory in result
    ] == [
        canonical.memory_id,
    ]

    assert all(
        memory.memory_id != consolidated.memory_id
        for memory in result
    )


def test_get_at_time_excludes_consolidated_memories(
    database_path,
):
    canonical = insert_memory(
        database_path,
        memory_id="mem_canonical",
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
        valid_to=datetime(
            2026,
            6,
            1,
            tzinfo=timezone.utc,
        ),
        status=MemoryStatus.HISTORICAL,
    )

    consolidated = insert_memory(
        database_path,
        memory_id="mem_consolidated",
        recorded_at=datetime(
            2026,
            1,
            2,
            tzinfo=timezone.utc,
        ),
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
        status=MemoryStatus.CONSOLIDATED,
        canonical_memory_id=canonical.memory_id,
    )

    result = get_at_time(
        canonical.memory_key,
        datetime(
            2026,
            5,
            31,
            tzinfo=timezone.utc,
        ),
        database_path,
    )

    assert result is not None
    assert result.memory_id == canonical.memory_id
    assert result.memory_id != consolidated.memory_id


def test_discard_marks_memory_discarded(database_path):
    memory = insert_memory(
        database_path,
        memory_id="mem_a",
        recorded_at=datetime(
            2026,
            1,
            1,
            tzinfo=timezone.utc,
        ),
        valid_to=None,
    )

    updated = discard(
        memory.memory_id,
        database_path,
    )

    assert updated.status == MemoryStatus.DISCARDED
    assert updated.valid_from == memory.valid_from
    assert updated.valid_to == memory.valid_to

    stored = get(
        memory.memory_id,
        database_path,
    )

    assert stored.status == MemoryStatus.DISCARDED


def test_discard_rejects_unknown_memory(database_path):
    with pytest.raises(MemoryNotFoundError):
        discard(
            "does_not_exist",
            database_path,
        )


def test_discard_is_idempotent_for_already_discarded_memory(
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
        valid_to=None,
        status=MemoryStatus.DISCARDED,
    )

    result = discard(
        memory.memory_id,
        database_path,
    )

    assert result.status == MemoryStatus.DISCARDED
    assert result.memory_id == memory.memory_id


def test_discard_rejects_consolidated_memory(
    database_path,
):
    memory = insert_memory(
        database_path,
        memory_id="mem_consolidated",
        recorded_at=datetime(
            2026,
            1,
            1,
            tzinfo=timezone.utc,
        ),
        valid_to=None,
        status=MemoryStatus.CONSOLIDATED,
        canonical_memory_id="mem_canonical",
    )

    with pytest.raises(InvalidConsolidationError):
        discard(
            memory.memory_id,
            database_path,
        )


def test_discarded_memory_excluded_from_get_current(
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
        valid_to=None,
    )

    discard(
        memory.memory_id,
        database_path,
    )

    assert get_current(
        memory.memory_key,
        database_path,
    ) is None


def make_hybrid_memory(
    memory_id: str,
    value: str,
    valid_from: str,
    valid_to: str | None = None,
):
    from app.memory.models import Memory

    return Memory(
        memory_id=memory_id,
        memory_key="user:language",
        subject="user",
        attribute="programming_language",
        value=value,
        memory_type=MemoryType.SKILL,
        valid_from=datetime.fromisoformat(
            valid_from.replace("Z", "+00:00")
        ),
        valid_to=(
            datetime.fromisoformat(
                valid_to.replace("Z", "+00:00")
            )
            if valid_to is not None
            else None
        ),
        precision=TimePrecision.EXACT,
        source_type=SourceType.CONVERSATION,
        source_id=f"conversation-{memory_id}",
        evidence_type=EvidenceType.EXPLICIT,
        confidence=0.95,
    )


def test_hybrid_search_at_time():
    memories = [
        make_hybrid_memory(
            "node",
            "Node.js",
            "2026-01-01T00:00:00Z",
            "2026-06-01T00:00:00Z",
        ),
        make_hybrid_memory(
            "go",
            "Go",
            "2026-06-01T00:00:00Z",
            "2026-09-01T00:00:00Z",
        ),
        make_hybrid_memory(
            "python",
            "Python",
            "2026-09-01T00:00:00Z",
        ),
    ]

    result = hybrid_search(
        memories=memories,
        semantic_scores={
            "node": 0.91,
            "go": 0.84,
            "python": 0.78,
        },
        at=datetime(
            2026,
            3,
            1,
            tzinfo=timezone.utc,
        ),
    )

    assert [
        candidate.memory.memory_id
        for candidate in result.candidates
    ] == ["node"]

    assert result.candidates[0].hybrid_score == 0.91
    assert result.conflicts == []


def test_hybrid_search_range():
    memories = [
        make_hybrid_memory(
            "node",
            "Node.js",
            "2026-01-01T00:00:00Z",
            "2026-06-01T00:00:00Z",
        ),
        make_hybrid_memory(
            "go",
            "Go",
            "2026-06-01T00:00:00Z",
            "2026-09-01T00:00:00Z",
        ),
        make_hybrid_memory(
            "python",
            "Python",
            "2026-09-01T00:00:00Z",
        ),
    ]

    result = hybrid_search(
        memories=memories,
        semantic_scores={
            "node": 0.84,
            "go": 0.93,
            "python": 0.78,
        },
        start=datetime(
            2026,
            5,
            1,
            tzinfo=timezone.utc,
        ),
        end=datetime(
            2026,
            7,
            1,
            tzinfo=timezone.utc,
        ),
    )

    assert [
        candidate.memory.memory_id
        for candidate in result.candidates
    ] == [
        "go",
        "node",
    ]


def test_hybrid_search_applies_top_k():
    memories = [
        make_hybrid_memory(
            "node",
            "Node.js",
            "2026-01-01T00:00:00Z",
        ),
        make_hybrid_memory(
            "python",
            "Python",
            "2026-01-01T00:00:00Z",
        ),
        make_hybrid_memory(
            "go",
            "Go",
            "2026-01-01T00:00:00Z",
        ),
    ]

    result = hybrid_search(
        memories=memories,
        semantic_scores={
            "node": 0.80,
            "python": 0.95,
            "go": 0.88,
        },
        at=datetime(
            2026,
            3,
            1,
            tzinfo=timezone.utc,
        ),
        top_k=2,
    )

    assert [
        candidate.memory.memory_id
        for candidate in result.candidates
    ] == [
        "python",
        "go",
    ]


def test_hybrid_search_propagates_conflicts():
    memories = [
        make_hybrid_memory(
            "python",
            "Python",
            "2026-01-01T00:00:00Z",
            "2026-06-01T00:00:00Z",
        ),
        make_hybrid_memory(
            "java",
            "Java",
            "2026-03-01T00:00:00Z",
            "2026-09-01T00:00:00Z",
        ),
    ]

    result = hybrid_search(
        memories=memories,
        semantic_scores={
            "python": 0.82,
            "java": 0.91,
        },
        at=datetime(
            2026,
            4,
            1,
            tzinfo=timezone.utc,
        ),
    )

    assert len(result.conflicts) == 1

    assert result.conflicts[0].memory_ids == (
        "java",
        "python",
    )


def test_hybrid_search_rejects_at_with_range():
    memories = []

    with pytest.raises(
        ValueError,
        match="either at or start/end",
    ):
        hybrid_search(
            memories=memories,
            semantic_scores={},
            at=datetime(
                2026,
                3,
                1,
                tzinfo=timezone.utc,
            ),
            start=datetime(
                2026,
                3,
                1,
                tzinfo=timezone.utc,
            ),
            end=datetime(
                2026,
                4,
                1,
                tzinfo=timezone.utc,
            ),
        )


def test_hybrid_search_requires_complete_range():
    memories = []

    with pytest.raises(
        ValueError,
        match="start and end must be provided together",
    ):
        hybrid_search(
            memories=memories,
            semantic_scores={},
            start=datetime(
                2026,
                3,
                1,
                tzinfo=timezone.utc,
            ),
        )


def test_hybrid_search_without_temporal_constraint():
    memories = [
        make_hybrid_memory(
            "node",
            "Node.js",
            "2026-01-01T00:00:00Z",
        ),
        make_hybrid_memory(
            "python",
            "Python",
            "2026-01-01T00:00:00Z",
        ),
    ]

    result = hybrid_search(
        memories=memories,
        semantic_scores={
            "node": 0.82,
            "python": 0.94,
        },
    )

    assert [
        candidate.memory.memory_id
        for candidate in result.candidates
    ] == [
        "python",
        "node",
    ]