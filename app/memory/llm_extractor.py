from abc import ABC, abstractmethod

from app.memory.extraction import MemoryExtractionResult
from app.memory.extractor import MemoryExtractor


class StructuredLLM(ABC):
    @abstractmethod
    def generate_structured(
        self,
        prompt: str,
        *,
        response_model: type[MemoryExtractionResult],
    ) -> MemoryExtractionResult:
        raise NotImplementedError


class LLMMemoryExtractor(MemoryExtractor):
    def __init__(self, llm: StructuredLLM):
        self.llm = llm

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

        prompt = self._build_prompt(
            conversation,
            source_id=source_id,
        )

        result = self.llm.generate_structured(
            prompt,
            response_model=MemoryExtractionResult,
        )

        return self._normalize_result(
            result,
            source_id=source_id,
        )

    @staticmethod
    def _build_prompt(
        conversation: str,
        *,
        source_id: str,
    ) -> str:
        return f"""
Extract durable user memories from the conversation below.

Only extract information that could be useful in future
conversations.

For every memory identify:
- subject
- attribute
- value
- memory type
- validity interval
- confidence
- whether the evidence is explicit or inferred

Do not invent facts.

Source ID:
{source_id}

Conversation:
{conversation}
""".strip()

    @staticmethod
    def _normalize_result(
        result: MemoryExtractionResult,
        *,
        source_id: str,
    ) -> MemoryExtractionResult:
        normalized = []

        for memory in result.memories:
            if memory.source_id != source_id:
                normalized.append(
                    memory.model_copy(
                        update={
                            "source_id": source_id,
                        }
                    )
                )
            else:
                normalized.append(memory)

        return MemoryExtractionResult(
            memories=normalized
        )