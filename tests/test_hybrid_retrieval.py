from datetime import datetime, timezone

from app.memory.models import (
    EvidenceType,
    Memory,
    MemoryStatus,
    MemoryType,
    SourceType,
)
from app.retrieval.conflict_detector import (
    ConflictDetector,
)
from app.retrieval.hybrid_ranker import (
    HybridRanker,
)
from app.retrieval.hybrid_scoring import (
    HybridScorer,
)
from app.retrieval.temporal_filter import (
    TemporalCandidateFilter,
)


def make_memory(
    memory_id: str,
    value: str,
    valid_from: str,
    valid_to: str | None = None,
    memory_key: str = "user.programming_language",
    status: MemoryStatus = MemoryStatus.ACTIVE,
) -> Memory:
    return Memory(
        memory_id=memory_id,
        memory_key=memory_key,
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
        status=status,
    )


def test_temporal_filter_then_semantic_ranking():
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

    temporal_filter = TemporalCandidateFilter()
    scorer = HybridScorer()
    ranker = HybridRanker()

    candidates = temporal_filter.filter_at_time(
        memories,
        datetime(
            2026,
            3,
            1,
            tzinfo=timezone.utc,
        ),
    )

    scored = scorer.score(
        [
            (candidates[0], 0.91),
        ]
    )

    ranked = ranker.rank(
        scored,
        top_k=1,
    )

    assert [
        result.memory.value
        for result in ranked
    ] == ["Node.js"]

    assert ranked[0].hybrid_score == 0.91


def test_temporal_filter_removes_future_memories_before_ranking():
    memories = [
        make_memory(
            "node",
            "Node.js",
            "2026-01-01T00:00:00Z",
            "2026-06-01T00:00:00Z",
        ),
        make_memory(
            "python",
            "Python",
            "2026-09-01T00:00:00Z",
        ),
    ]

    temporal_filter = TemporalCandidateFilter()

    candidates = temporal_filter.filter_at_time(
        memories,
        datetime(
            2026,
            3,
            1,
            tzinfo=timezone.utc,
        ),
    )

    assert [
        memory.memory_id
        for memory in candidates
    ] == ["node"]


def test_semantic_scores_rank_temporally_valid_candidates():
    memories = [
        make_memory(
            "node",
            "Node.js",
            "2026-01-01T00:00:00Z",
            "2026-06-01T00:00:00Z",
        ),
        make_memory(
            "express",
            "Express.js",
            "2026-01-01T00:00:00Z",
            "2026-06-01T00:00:00Z",
        ),
        make_memory(
            "python",
            "Python",
            "2026-09-01T00:00:00Z",
        ),
    ]

    temporal_filter = TemporalCandidateFilter()
    scorer = HybridScorer()
    ranker = HybridRanker()

    candidates = temporal_filter.filter_at_time(
        memories,
        datetime(
            2026,
            3,
            1,
            tzinfo=timezone.utc,
        ),
    )

    scored = scorer.score(
        [
            (candidates[0], 0.84),
            (candidates[1], 0.93),
        ]
    )

    ranked = ranker.rank(scored)

    assert [
        result.memory.value
        for result in ranked
    ] == [
        "Express.js",
        "Node.js",
    ]


def test_conflict_is_detected_among_temporally_valid_memories():
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

    temporal_filter = TemporalCandidateFilter()
    detector = ConflictDetector()

    candidates = temporal_filter.filter_at_time(
        memories,
        datetime(
            2026,
            4,
            1,
            tzinfo=timezone.utc,
        ),
    )

    conflicts = detector.detect(candidates)

    assert len(conflicts) == 1
    assert conflicts[0].memory_ids == (
        "java",
        "python",
    )


def test_non_overlapping_memories_are_not_conflict_at_time():
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
            "2026-06-01T00:00:00Z",
            "2026-09-01T00:00:00Z",
        ),
    ]

    temporal_filter = TemporalCandidateFilter()
    detector = ConflictDetector()

    candidates = temporal_filter.filter_at_time(
        memories,
        datetime(
            2026,
            3,
            1,
            tzinfo=timezone.utc,
        ),
    )

    conflicts = detector.detect(candidates)

    assert conflicts == []


def test_top_k_limits_final_results():
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

    temporal_filter = TemporalCandidateFilter()
    scorer = HybridScorer()
    ranker = HybridRanker()

    candidates = temporal_filter.filter_at_time(
        memories,
        datetime(
            2026,
            3,
            1,
            tzinfo=timezone.utc,
        ),
    )

    scored = scorer.score(
        [
            (candidates[0], 0.80),
            (candidates[1], 0.95),
            (candidates[2], 0.88),
        ]
    )

    ranked = ranker.rank(
        scored,
        top_k=2,
    )

    assert [
        result.memory.memory_id
        for result in ranked
    ] == [
        "python",
        "go",
    ]


def test_discarded_memory_does_not_enter_hybrid_pipeline():
    memories = [
        make_memory(
            "python",
            "Python",
            "2026-01-01T00:00:00Z",
            status=MemoryStatus.DISCARDED,
        ),
        make_memory(
            "node",
            "Node.js",
            "2026-01-01T00:00:00Z",
        ),
    ]

    temporal_filter = TemporalCandidateFilter()

    candidates = temporal_filter.filter_at_time(
        memories,
        datetime(
            2026,
            3,
            1,
            tzinfo=timezone.utc,
        ),
    )

    assert [
        memory.memory_id
        for memory in candidates
    ] == ["node"]


def test_range_filter_then_conflict_detection():
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
        make_memory(
            "go",
            "Go",
            "2026-10-01T00:00:00Z",
        ),
    ]

    temporal_filter = TemporalCandidateFilter()
    detector = ConflictDetector()

    candidates = temporal_filter.filter_range(
        memories,
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

    conflicts = detector.detect(candidates)

    assert [
        memory.memory_id
        for memory in candidates
    ] == [
        "python",
        "java",
    ]

    assert len(conflicts) == 1
    assert conflicts[0].memory_ids == (
        "java",
        "python",
    )