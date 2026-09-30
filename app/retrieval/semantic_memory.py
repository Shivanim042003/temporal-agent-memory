from dataclasses import dataclass
from pathlib import Path

from app.memory.models import Memory
from app.retrieval.embeddings import EmbeddingModel
from app.retrieval.faiss_index import FAISSIndex
from app.retrieval.id_mapping import MemoryIDMapping
from app.storage.memory_repository import get


@dataclass(frozen=True)
class SemanticMemoryResult:
    memory: Memory
    score: float


class SemanticMemoryIndex:
    def __init__(
        self,
        embedding_model: EmbeddingModel,
        faiss_index: FAISSIndex,
        id_mapping: MemoryIDMapping,
    ):
        self.embedding_model = embedding_model
        self.faiss_index = faiss_index
        self.id_mapping = id_mapping

    @property
    def size(self) -> int:
        return self.faiss_index.size

    def add(self, memory: Memory) -> int:
        embedding = self.embedding_model.embed_memory(
            memory
        )

        vector_id = self.faiss_index.add(
            embedding
        )

        self.id_mapping.add(
            vector_id=vector_id,
            memory_id=memory.memory_id,
        )

        return vector_id

    def search(
        self,
        query: str,
        k: int,
        db_path: Path | str = "data/memory.db",
    ) -> list[SemanticMemoryResult]:
        if not query.strip():
            raise ValueError(
                "query must not be empty"
            )

        embedding = self.embedding_model.embed_text(
            query
        )

        scores, vector_ids = self.faiss_index.search(
            embedding,
            k,
        )

        results: list[SemanticMemoryResult] = []

        for score, vector_id in zip(
            scores,
            vector_ids,
        ):
            memory_id = self.id_mapping.get_memory_id(
                vector_id
            )

            if memory_id is None:
                continue

            memory = get(
                memory_id,
                db_path=db_path,
            )

            if memory is None:
                continue

            results.append(
                SemanticMemoryResult(
                    memory=memory,
                    score=score,
                )
            )

        return results

    def save(
        self,
        index_path: Path | str,
        mapping_path: Path | str,
    ) -> None:
        self.faiss_index.save(index_path)
        self.id_mapping.save(mapping_path)

    @classmethod
    def load(
        cls,
        embedding_model: EmbeddingModel,
        index_path: Path | str,
        mapping_path: Path | str,
    ) -> "SemanticMemoryIndex":
        faiss_index = FAISSIndex.load(
            index_path
        )

        id_mapping = MemoryIDMapping.load(
            mapping_path
        )

        if faiss_index.size != id_mapping.size:
            raise ValueError(
                "FAISS index size does not match "
                "memory ID mapping size"
            )

        return cls(
            embedding_model=embedding_model,
            faiss_index=faiss_index,
            id_mapping=id_mapping,
        )