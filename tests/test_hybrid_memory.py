from datetime import datetime, timezone

from app.memory.models import (
    EvidenceType,
    Memory,
    MemoryType,
    SourceType,
)
from app.retrieval.hybrid_memory import (
    HybridMemoryRetriever,
)


def make_memory(
    memory_id: str,
    value: str,
    valid_from: str,
    valid_to: str | None = None,
) -> Memory:
    return Memory(
        memory_id=memory_id,
        memory_key="user.programming_language",
        subject="user",
        attribute="programming_language",
        value=value,
        memory_type=MemoryType.SKILL,
        valid_from=valid_from,
        valid_to=valid_to,
        source_type=SourceType.CONVERSATION,
        source_id="conversation_1",
        evidence_type=EvidenceType.EXPLICIT,
        confidence=1.0,
    )


def test_search_at_time_filters_then_ranks():
    memories = [
        make_memory(
            "node",
            "Node.js",
            "2026-01-01T00:00:00Z",
            "2026-06-01T00:00:00Z",
        ),
        make_memory(
            "go",
            "Go",
            "2026-06-01T00:00:00Z",
            "2026-09-01T00:00:00Z",
        ),
        make_memory(
            "python",
            "Python",
            "2026-09-01T00:00:00Z",
        ),
    ]

    retriever = HybridMemoryRetriever()

    result = retriever.search_at_time(
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


def test_search_at_time_returns_conflicts():
    memories = [
        make_memory(
            "python",
            "Python",
            "2026-01-01T00:00:00Z",
            "2026-06-01T00:00:00Z",
        ),
        make_memory(
            "java",
            "Java",
            "2026-03-01T00:00:00Z",
            "2026-09-01T00:00:00Z",
        ),
    ]

    retriever = HybridMemoryRetriever()

    result = retriever.search_at_time(
        memories=memories,
        semantic_scores={
            "python": 0.88,
            "java": 0.92,
        },
        at=datetime(
            2026,
            4,
            1,
            tzinfo=timezone.utc,
        ),
    )

    assert [
        candidate.memory.memory_id
        for candidate in result.candidates
    ] == [
        "java",
        "python",
    ]

    assert len(result.conflicts) == 1

    assert result.conflicts[0].memory_ids == (
        "java",
        "python",
    )


def test_search_at_time_applies_top_k():
    memories = [
        make_memory(
            "node",
            "Node.js",
            "2026-01-01T00:00:00Z",
        ),
        make_memory(
            "python",
            "Python",
            "2026-01-01T00:00:00Z",
        ),
        make_memory(
            "go",
            "Go",
            "2026-01-01T00:00:00Z",
        ),
    ]

    retriever = HybridMemoryRetriever()

    result = retriever.search_at_time(
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


def test_search_at_time_ignores_missing_semantic_scores():
    memories = [
        make_memory(
            "node",
            "Node.js",
            "2026-01-01T00:00:00Z",
        ),
        make_memory(
            "python",
            "Python",
            "2026-01-01T00:00:00Z",
        ),
    ]

    retriever = HybridMemoryRetriever()

    result = retriever.search_at_time(
        memories=memories,
        semantic_scores={
            "node": 0.90,
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


def test_search_range_filters_and_ranks():
    memories = [
        make_memory(
            "node",
            "Node.js",
            "2026-01-01T00:00:00Z",
            "2026-06-01T00:00:00Z",
        ),
        make_memory(
            "go",
            "Go",
            "2026-06-01T00:00:00Z",
            "2026-09-01T00:00:00Z",
        ),
        make_memory(
            "python",
            "Python",
            "2026-09-01T00:00:00Z",
        ),
    ]

    retriever = HybridMemoryRetriever()

    result = retriever.search_range(
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


def test_search_range_returns_conflicts():
    memories = [
        make_memory(
            "python",
            "Python",
            "2026-01-01T00:00:00Z",
            "2026-06-01T00:00:00Z",
        ),
        make_memory(
            "java",
            "Java",
            "2026-04-01T00:00:00Z",
            "2026-09-01T00:00:00Z",
        ),
    ]

    retriever = HybridMemoryRetriever()

    result = retriever.search_range(
        memories=memories,
        semantic_scores={
            "python": 0.82,
            "java": 0.91,
        },
        start=datetime(
            2026,
            3,
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

    assert len(result.conflicts) == 1

    assert result.conflicts[0].memory_ids == (
        "java",
        "python",
    )


def test_search_at_time_with_no_semantic_scores_returns_no_candidates():
    memories = [
        make_memory(
            "python",
            "Python",
            "2026-01-01T00:00:00Z",
        )
    ]

    retriever = HybridMemoryRetriever()

    result = retriever.search_at_time(
        memories=memories,
        semantic_scores={},
        at=datetime(
            2026,
            3,
            1,
            tzinfo=timezone.utc,
        ),
    )

    assert result.candidates == []
    assert result.conflicts == []


def test_search_at_time_uses_deterministic_ranking():
    memories = [
        make_memory(
            "memory_b",
            "Python",
            "2026-01-01T00:00:00Z",
        ),
        make_memory(
            "memory_a",
            "Python",
            "2026-01-01T00:00:00Z",
        ),
    ]

    retriever = HybridMemoryRetriever()

    result = retriever.search_at_time(
        memories=memories,
        semantic_scores={
            "memory_b": 0.90,
            "memory_a": 0.90,
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
    ] == [
        "memory_a",
        "memory_b",
    ]