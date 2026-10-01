from collections.abc import Callable

from app.memory.extraction import MemoryExtractionResult
from app.memory.extractor import MemoryExtractor
from app.memory.models import Memory
from app.storage.memory_repository import insert


class MemoryIngestionService:
    def __init__(
        self,
        extractor: MemoryExtractor,
        id_factory: Callable[[], str],
    ):
        self.extractor = extractor
        self.id_factory = id_factory

    def ingest(
        self,
        conversation: str,
        *,
        source_id: str,
        db_path,
    ) -> list[Memory]:
        extraction_result = self.extractor.extract(
            conversation,
            source_id=source_id,
        )

        memories = [
            candidate.to_memory(
                memory_id=self.id_factory(),
            )
            for candidate in extraction_result.memories
        ]

        for memory in memories:
            insert(
                memory,
                db_path,
            )

        return memories