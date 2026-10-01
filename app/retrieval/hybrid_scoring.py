from dataclasses import dataclass

from app.memory.models import Memory


@dataclass(frozen=True)
class HybridCandidate:
    memory: Memory
    semantic_score: float
    hybrid_score: float


class HybridScorer:
    def score(
        self,
        candidates: list[tuple[Memory, float]],
    ) -> list[HybridCandidate]:
        results: list[HybridCandidate] = []

        for memory, semantic_score in candidates:
            if not 0.0 <= semantic_score <= 1.0:
                raise ValueError(
                    "semantic_score must be between 0 and 1"
                )

            results.append(
                HybridCandidate(
                    memory=memory,
                    semantic_score=semantic_score,
                    hybrid_score=semantic_score,
                )
            )

        return results
    