from datetime import datetime, timezone

import pytest

from app.memory.extraction import (
    ExtractionEvidence,
    MemoryExtractionCandidate,
    MemoryExtractionResult,
)
from app.memory.llm_extractor import (
    LLMMemoryExtractor,
    StructuredLLM,
)
from app.memory.models import MemoryType


class FakeStructuredLLM(StructuredLLM):
    def __init__(
        self,
        result: MemoryExtractionResult,
    ):
        self.result = result
        self.last_prompt = None
        self.last_response_model = None

    def generate_structured(
        self,
        prompt: str,
        *,
        response_model: type[MemoryExtractionResult],
    ) -> MemoryExtractionResult:
        self.last_prompt = prompt
        self.last_response_model = response_model

        return self.result


def make_candidate(
    *,
    source_id: str = "conversation-1",
    value: str = "Python",
):
    return MemoryExtractionCandidate(
        subject="user",
        attribute="programming_language",
        value=value,
        memory_type=MemoryType.SKILL,
        valid_from=datetime(
            2026,
            9,
            1,
            tzinfo=timezone.utc,
        ),
        confidence=0.95,
        evidence=ExtractionEvidence.EXPLICIT,
        source_id=source_id,
    )


def test_llm_extractor_calls_structured_llm():
    expected = MemoryExtractionResult(
        memories=[
            make_candidate()
        ]
    )

    llm = FakeStructuredLLM(expected)

    extractor = LLMMemoryExtractor(llm)

    result = extractor.extract(
        "I am currently learning Python.",
        source_id="conversation-1",
    )

    assert result == expected
    assert llm.last_response_model is MemoryExtractionResult


def test_llm_extractor_builds_prompt_with_conversation():
    llm = FakeStructuredLLM(
        MemoryExtractionResult()
    )

    extractor = LLMMemoryExtractor(llm)

    extractor.extract(
        "I am learning Python.",
        source_id="conversation-42",
    )

    assert llm.last_prompt is not None
    assert "I am learning Python." in llm.last_prompt
    assert "conversation-42" in llm.last_prompt
    assert "Extract durable user memories" in llm.last_prompt


def test_llm_extractor_normalizes_source_id():
    result = MemoryExtractionResult(
        memories=[
            make_candidate(
                source_id="wrong-source"
            )
        ]
    )

    llm = FakeStructuredLLM(result)

    extractor = LLMMemoryExtractor(llm)

    extracted = extractor.extract(
        "I am learning Python.",
        source_id="conversation-99",
    )

    assert len(extracted.memories) == 1
    assert (
        extracted.memories[0].source_id
        == "conversation-99"
    )


def test_llm_extractor_preserves_correct_source_id():
    result = MemoryExtractionResult(
        memories=[
            make_candidate(
                source_id="conversation-99"
            )
        ]
    )

    llm = FakeStructuredLLM(result)

    extractor = LLMMemoryExtractor(llm)

    extracted = extractor.extract(
        "I am learning Python.",
        source_id="conversation-99",
    )

    assert (
        extracted.memories[0].source_id
        == "conversation-99"
    )


def test_llm_extractor_rejects_empty_conversation():
    llm = FakeStructuredLLM(
        MemoryExtractionResult()
    )

    extractor = LLMMemoryExtractor(llm)

    with pytest.raises(
        ValueError,
        match="conversation must not be empty",
    ):
        extractor.extract(
            "   ",
            source_id="conversation-1",
        )


def test_llm_extractor_rejects_empty_source_id():
    llm = FakeStructuredLLM(
        MemoryExtractionResult()
    )

    extractor = LLMMemoryExtractor(llm)

    with pytest.raises(
        ValueError,
        match="source_id must not be empty",
    ):
        extractor.extract(
            "I am learning Python.",
            source_id="   ",
        )


def test_llm_extractor_supports_multiple_memories():
    result = MemoryExtractionResult(
        memories=[
            make_candidate(value="Python"),
            make_candidate(value="Java"),
        ]
    )

    llm = FakeStructuredLLM(result)

    extractor = LLMMemoryExtractor(llm)

    extracted = extractor.extract(
        "I have experience with Python and Java.",
        source_id="conversation-1",
    )

    assert len(extracted.memories) == 2
    assert [
        memory.value
        for memory in extracted.memories
    ] == [
        "Python",
        "Java",
    ]