from abc import ABC, abstractmethod

from app.memory.extraction import MemoryExtractionResult


class MemoryExtractor(ABC):
    @abstractmethod
    def extract(
        self,
        conversation: str,
        *,
        source_id: str,
    ) -> MemoryExtractionResult:
        """Extract structured memories from conversation text."""
        raise NotImplementedError


class StaticMemoryExtractor(MemoryExtractor):
    """
    Deterministic extractor used for development and testing.

    A real LLM-backed extractor will implement the same interface.
    """

    def __init__(
        self,
        result: MemoryExtractionResult | None = None,
    ):
        self.result = result or MemoryExtractionResult()

    def extract(
        self,
        conversation: str,
        *,
        source_id: str,
    ) -> MemoryExtractionResult:
        if not conversation.strip():
            raise ValueError(
                "conversation must not be empty"
            )

        if not source_id.strip():
            raise ValueError(
                "source_id must not be empty"
            )

        return self.result