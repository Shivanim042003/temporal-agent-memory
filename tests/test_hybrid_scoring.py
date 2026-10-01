from datetime import datetime, timezone

import pytest

from app.memory.models import (
    EvidenceType,
    Memory,
    MemoryType,
    SourceType,
)
from app.retrieval.hybrid_scoring import (
    HybridScorer,
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


@pytest.fixture
def scorer() -> HybridScorer:
    return HybridScorer()


def test_score_preserves_semantic_score(
    scorer,
):
    memory = make_memory("python")

    results = scorer.score(
        [(memory, 0.91)]
    )

    assert len(results) == 1
    assert results[0].memory == memory
    assert results[0].semantic_score == 0.91
    assert results[0].hybrid_score == 0.91


def test_score_handles_multiple_candidates(
    scorer,
):
    node = make_memory("node")
    go = make_memory("go")
    python = make_memory("python")

    results = scorer.score(
        [
            (node, 0.91),
            (go, 0.84),
            (python, 0.78),
        ]
    )

    assert [
        result.memory.memory_id
        for result in results
    ] == [
        "node",
        "go",
        "python",
    ]

    assert [
        result.hybrid_score
        for result in results
    ] == [
        0.91,
        0.84,
        0.78,
    ]


def test_score_accepts_zero(
    scorer,
):
    memory = make_memory("python")

    results = scorer.score(
        [(memory, 0.0)]
    )

    assert results[0].hybrid_score == 0.0


def test_score_accepts_one(
    scorer,
):
    memory = make_memory("python")

    results = scorer.score(
        [(memory, 1.0)]
    )

    assert results[0].hybrid_score == 1.0


def test_score_rejects_negative_score(
    scorer,
):
    memory = make_memory("python")

    with pytest.raises(
        ValueError,
        match="semantic_score must be between 0 and 1",
    ):
        scorer.score(
            [(memory, -0.1)]
        )


def test_score_rejects_score_above_one(
    scorer,
):
    memory = make_memory("python")

    with pytest.raises(
        ValueError,
        match="semantic_score must be between 0 and 1",
    ):
        scorer.score(
            [(memory, 1.1)]
        )


def test_score_empty_candidates_returns_empty(
    scorer,
):
    results = scorer.score([])

    assert results == []