from dataclasses import dataclass

from app.retrieval.faiss_index import FAISSIndex


@dataclass(frozen=True)
class SearchResult:
    vector_id: int
    score: float


class SemanticRetriever:
    def __init__(self, index: FAISSIndex):
        self.index = index

    def search(
        self,
        query_embedding: list[float],
        k: int,
    ) -> list[SearchResult]:
        scores, ids = self.index.search(
            query_embedding,
            k,
        )

        return [
            SearchResult(
                vector_id=vector_id,
                score=score,
            )
            for score, vector_id in zip(scores, ids)
        ]
    