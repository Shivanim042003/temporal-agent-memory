from dataclasses import dataclass
from datetime import datetime

from app.memory.models import Memory
from app.retrieval.conflict_detector import (
    ConflictDetector,
    ConflictGroup,
)
from app.retrieval.hybrid_ranker import (
    HybridRanker,
)
from app.retrieval.hybrid_scoring import (
    HybridCandidate,
    HybridScorer,
)
from app.retrieval.temporal_filter import (
    TemporalCandidateFilter,
)


@dataclass(frozen=True)
class HybridMemoryResult:
    candidates: list[HybridCandidate]
    conflicts: list[ConflictGroup]


class HybridMemoryRetriever:
    def __init__(
        self,
        temporal_filter: TemporalCandidateFilter | None = None,
        scorer: HybridScorer | None = None,
        conflict_detector: ConflictDetector | None = None,
        ranker: HybridRanker | None = None,
    ):
        self.temporal_filter = (
            temporal_filter
            or TemporalCandidateFilter()
        )

        self.scorer = (
            scorer
            or HybridScorer()
        )

        self.conflict_detector = (
            conflict_detector
            or ConflictDetector()
        )

        self.ranker = (
            ranker
            or HybridRanker()
        )

    def search_at_time(
        self,
        memories: list[Memory],
        semantic_scores: dict[str, float],
        at: datetime,
        top_k: int | None = None,
    ) -> HybridMemoryResult:
        temporal_candidates = (
            self.temporal_filter.filter_at_time(
                memories,
                at,
            )
        )

        scored_candidates = self._score_candidates(
            temporal_candidates,
            semantic_scores,
        )

        conflicts = self.conflict_detector.detect(
            temporal_candidates
        )

        ranked_candidates = self.ranker.rank(
            scored_candidates,
            top_k=top_k,
        )

        return HybridMemoryResult(
            candidates=ranked_candidates,
            conflicts=conflicts,
        )

    def search_range(
        self,
        memories: list[Memory],
        semantic_scores: dict[str, float],
        start: datetime,
        end: datetime,
        top_k: int | None = None,
    ) -> HybridMemoryResult:
        temporal_candidates = (
            self.temporal_filter.filter_range(
                memories,
                start,
                end,
            )
        )

        scored_candidates = self._score_candidates(
            temporal_candidates,
            semantic_scores,
        )

        conflicts = self.conflict_detector.detect(
            temporal_candidates
        )

        ranked_candidates = self.ranker.rank(
            scored_candidates,
            top_k=top_k,
        )

        return HybridMemoryResult(
            candidates=ranked_candidates,
            conflicts=conflicts,
        )

    def _score_candidates(
        self,
        memories: list[Memory],
        semantic_scores: dict[str, float],
    ) -> list[HybridCandidate]:
        candidates: list[tuple[Memory, float]] = []

        for memory in memories:
            if memory.memory_id not in semantic_scores:
                continue

            candidates.append(
                (
                    memory,
                    semantic_scores[memory.memory_id],
                )
            )

        return self.scorer.score(candidates)