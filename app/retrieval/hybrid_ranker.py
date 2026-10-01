from app.retrieval.hybrid_scoring import HybridCandidate


class HybridRanker:
    def rank(
        self,
        candidates: list[HybridCandidate],
        top_k: int | None = None,
    ) -> list[HybridCandidate]:
        if top_k is not None and top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero"
            )

        ranked = sorted(
            candidates,
            key=lambda candidate: (
                -candidate.hybrid_score,
                candidate.memory.memory_id,
            ),
        )

        if top_k is not None:
            return ranked[:top_k]

        return ranked