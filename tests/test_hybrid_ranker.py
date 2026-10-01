from datetime import datetime, timezone

import pytest

from app.memory.models import (
    EvidenceType,
    Memory,
    MemoryType,
    SourceType,
)
from app.retrieval.hybrid_ranker import (
    HybridRanker,
)
from app.retrieval.hybrid_scoring import (
    HybridCandidate,
)


def make_memory(memory_id: str) -> Memory:
    return Memory(
        memory_id=memory_id,
        memory_key=f"user.language.{memory_id}",
        subject="user",
        attribute="programming_language",
        value=memory_id,
        memory_type=MemoryType.SKILL,
        valid_from=datetime(
            2026,
            1,
            1,
            tzinfo=timezone.utc,
        ),
        source_type=SourceType.CONVERSATION,
        source_id="conversation_1",
        evidence_type=EvidenceType.EXPLICIT,
        confidence=1.0,
    )


def make_candidate(
    memory_id: str,
    score: float,
) -> HybridCandidate:
    return HybridCandidate(
        memory=make_memory(memory_id),
        semantic_score=score,
        hybrid_score=score,
    )


@pytest.fixture
def ranker() -> HybridRanker:
    return HybridRanker()


def test_rank_orders_by_highest_score_first(
    ranker,
):
    candidates = [
        make_candidate("python", 0.78),
        make_candidate("node", 0.91),
        make_candidate("go", 0.84),
    ]

    results = ranker.rank(candidates)

    assert [
        candidate.memory.memory_id
        for candidate in results
    ] == [
        "node",
        "go",
        "python",
    ]


def test_rank_preserves_scores(
    ranker,
):
    candidates = [
        make_candidate("python", 0.78),
        make_candidate("node", 0.91),
    ]

    results = ranker.rank(candidates)

    assert [
        candidate.hybrid_score
        for candidate in results
    ] == [
        0.91,
        0.78,
    ]


def test_rank_applies_top_k(
    ranker,
):
    candidates = [
        make_candidate("python", 0.78),
        make_candidate("node", 0.91),
        make_candidate("go", 0.84),
    ]

    results = ranker.rank(
        candidates,
        top_k=2,
    )

    assert [
        candidate.memory.memory_id
        for candidate in results
    ] == [
        "node",
        "go",
    ]


def test_rank_top_k_larger_than_candidates(
    ranker,
):
    candidates = [
        make_candidate("python", 0.78),
        make_candidate("node", 0.91),
    ]

    results = ranker.rank(
        candidates,
        top_k=10,
    )

    assert [
        candidate.memory.memory_id
        for candidate in results
    ] == [
        "node",
        "python",
    ]


def test_rank_uses_memory_id_as_tie_breaker(
    ranker,
):
    candidates = [
        make_candidate("memory_b", 0.90),
        make_candidate("memory_a", 0.90),
        make_candidate("memory_c", 0.80),
    ]

    results = ranker.rank(candidates)

    assert [
        candidate.memory.memory_id
        for candidate in results
    ] == [
        "memory_a",
        "memory_b",
        "memory_c",
    ]


def test_rank_does_not_mutate_input(
    ranker,
):
    candidates = [
        make_candidate("python", 0.78),
        make_candidate("node", 0.91),
    ]

    original_order = [
        candidate.memory.memory_id
        for candidate in candidates
    ]

    ranker.rank(candidates)

    assert [
        candidate.memory.memory_id
        for candidate in candidates
    ] == original_order


def test_rank_empty_candidates(
    ranker,
):
    results = ranker.rank([])

    assert results == []


def test_rank_rejects_zero_top_k(
    ranker,
):
    with pytest.raises(
        ValueError,
        match="top_k must be greater than zero",
    ):
        ranker.rank(
            [],
            top_k=0,
        )


def test_rank_rejects_negative_top_k(
    ranker,
):
    with pytest.raises(
        ValueError,
        match="top_k must be greater than zero",
    ):
        ranker.rank(
            [],
            top_k=-1,
        )