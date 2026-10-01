from datetime import datetime, timezone

import pytest

from app.memory.extraction import (
    ExtractionEvidence,
    MemoryExtractionCandidate,
    MemoryExtractionResult,
)
from app.memory.extractor import (
    MemoryExtractor,
    StaticMemoryExtractor,
)
from app.memory.models import MemoryType


def make_candidate(
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
        source_id="conversation-1",
    )


def test_static_extractor_implements_memory_extractor():
    extractor = StaticMemoryExtractor()

    assert isinstance(
        extractor,
        MemoryExtractor,
    )


def test_static_extractor_returns_empty_result():
    extractor = StaticMemoryExtractor()

    result = extractor.extract(
        "I am learning Python.",
        source_id="conversation-1",
    )

    assert isinstance(
        result,
        MemoryExtractionResult,
    )

    assert result.memories == []


def test_static_extractor_returns_configured_result():
    candidate = make_candidate()

    expected = MemoryExtractionResult(
        memories=[candidate]
    )

    extractor = StaticMemoryExtractor(
        result=expected
    )

    result = extractor.extract(
        "I am learning Python.",
        source_id="conversation-1",
    )

    assert result == expected
    assert result.memories[0].value == "Python"


def test_static_extractor_rejects_empty_conversation():
    extractor = StaticMemoryExtractor()

    with pytest.raises(
        ValueError,
        match="conversation must not be empty",
    ):
        extractor.extract(
            "   ",
            source_id="conversation-1",
        )


def test_static_extractor_rejects_empty_source_id():
    extractor = StaticMemoryExtractor()

    with pytest.raises(
        ValueError,
        match="source_id must not be empty",
    ):
        extractor.extract(
            "I am learning Python.",
            source_id="   ",
        )


def test_static_extractor_preserves_multiple_memories():
    result = MemoryExtractionResult(
        memories=[
            make_candidate("Python"),
            make_candidate("Java"),
        ]
    )

    extractor = StaticMemoryExtractor(
        result=result
    )

    extracted = extractor.extract(
        "I have used Python and Java.",
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