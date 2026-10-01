from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from app.memory.extraction import (
    ExtractionEvidence,
    MemoryExtractionCandidate,
    MemoryExtractionResult,
)
from app.memory.models import (
    EvidenceType,
    MemoryType,
    SourceType,
    TimePrecision,
)


def test_extraction_candidate_accepts_valid_memory():
    candidate = MemoryExtractionCandidate(
        subject="user",
        attribute="programming_language",
        value="Python",
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

    assert candidate.subject == "user"
    assert candidate.attribute == "programming_language"
    assert candidate.value == "Python"
    assert candidate.memory_type == MemoryType.SKILL
    assert candidate.confidence == 0.95


def test_extraction_candidate_normalizes_datetime_to_utc():
    candidate = MemoryExtractionCandidate(
        subject="user",
        attribute="programming_language",
        value="Python",
        memory_type=MemoryType.SKILL,
        valid_from=datetime(
            2026,
            9,
            1,
            12,
            tzinfo=timezone.utc,
        ),
        confidence=1.0,
        evidence=ExtractionEvidence.EXPLICIT,
        source_id="conversation-1",
    )

    assert candidate.valid_from.tzinfo == timezone.utc


def test_extraction_candidate_rejects_naive_datetime():
    with pytest.raises(
        ValidationError,
        match="datetime must be timezone-aware",
    ):
        MemoryExtractionCandidate(
            subject="user",
            attribute="programming_language",
            value="Python",
            memory_type=MemoryType.SKILL,
            valid_from=datetime(
                2026,
                9,
                1,
            ),
            confidence=1.0,
            evidence=ExtractionEvidence.EXPLICIT,
            source_id="conversation-1",
        )


def test_extraction_candidate_rejects_invalid_confidence():
    with pytest.raises(ValidationError):
        MemoryExtractionCandidate(
            subject="user",
            attribute="programming_language",
            value="Python",
            memory_type=MemoryType.SKILL,
            valid_from=datetime(
                2026,
                9,
                1,
                tzinfo=timezone.utc,
            ),
            confidence=1.5,
            evidence=ExtractionEvidence.EXPLICIT,
            source_id="conversation-1",
        )


def test_extraction_candidate_rejects_invalid_interval():
    with pytest.raises(
        ValidationError,
        match="valid_to must be later than valid_from",
    ):
        MemoryExtractionCandidate(
            subject="user",
            attribute="programming_language",
            value="Python",
            memory_type=MemoryType.SKILL,
            valid_from=datetime(
                2026,
                9,
                1,
                tzinfo=timezone.utc,
            ),
            valid_to=datetime(
                2026,
                8,
                1,
                tzinfo=timezone.utc,
            ),
            confidence=1.0,
            evidence=ExtractionEvidence.EXPLICIT,
            source_id="conversation-1",
        )


def test_extraction_result_defaults_to_empty():
    result = MemoryExtractionResult()

    assert result.memories == []


def test_extraction_result_accepts_multiple_memories():
    first = MemoryExtractionCandidate(
        subject="user",
        attribute="programming_language",
        value="Python",
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

    second = MemoryExtractionCandidate(
        subject="user",
        attribute="goal",
        value="Learn LLM engineering",
        memory_type=MemoryType.GOAL,
        valid_from=datetime(
            2026,
            9,
            1,
            tzinfo=timezone.utc,
        ),
        confidence=0.90,
        evidence=ExtractionEvidence.INFERRED,
        source_id="conversation-1",
    )

    result = MemoryExtractionResult(
        memories=[first, second]
    )

    assert len(result.memories) == 2
    assert result.memories[0].value == "Python"
    assert result.memories[1].value == "Learn LLM engineering"


def test_to_memory_maps_explicit_evidence():
    candidate = MemoryExtractionCandidate(
        subject="user",
        attribute="programming_language",
        value="Python",
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

    memory = candidate.to_memory(
        memory_id="memory-1"
    )

    assert memory.memory_id == "memory-1"
    assert memory.memory_key == "user:programming_language"
    assert memory.subject == "user"
    assert memory.attribute == "programming_language"
    assert memory.value == "Python"
    assert memory.memory_type == MemoryType.SKILL
    assert memory.evidence_type == EvidenceType.EXPLICIT
    assert memory.source_type == SourceType.CONVERSATION
    assert memory.source_id == "conversation-1"
    assert memory.confidence == 0.95
    assert memory.status.value == "active"


def test_to_memory_maps_inferred_evidence():
    candidate = MemoryExtractionCandidate(
        subject="user",
        attribute="goal",
        value="Learn LLM engineering",
        memory_type=MemoryType.GOAL,
        valid_from=datetime(
            2026,
            9,
            1,
            tzinfo=timezone.utc,
        ),
        confidence=0.80,
        evidence=ExtractionEvidence.INFERRED,
        source_id="conversation-2",
    )

    memory = candidate.to_memory(
        memory_id="memory-2"
    )

    assert memory.evidence_type == EvidenceType.INFERRED
    assert memory.memory_type == MemoryType.GOAL
    assert memory.value == "Learn LLM engineering"


def test_to_memory_preserves_temporal_interval():
    valid_from = datetime(
        2026,
        1,
        1,
        tzinfo=timezone.utc,
    )

    valid_to = datetime(
        2026,
        6,
        1,
        tzinfo=timezone.utc,
    )

    candidate = MemoryExtractionCandidate(
        subject="user",
        attribute="programming_language",
        value="Node.js",
        memory_type=MemoryType.SKILL,
        valid_from=valid_from,
        valid_to=valid_to,
        confidence=0.90,
        evidence=ExtractionEvidence.EXPLICIT,
        source_id="conversation-3",
    )

    memory = candidate.to_memory(
        memory_id="memory-3",
        precision=TimePrecision.MONTH,
    )

    assert memory.valid_from == valid_from
    assert memory.valid_to == valid_to
    assert memory.precision == TimePrecision.MONTH


def test_to_memory_supports_custom_source_type():
    candidate = MemoryExtractionCandidate(
        subject="user",
        attribute="goal",
        value="Build an AI system",
        memory_type=MemoryType.GOAL,
        valid_from=datetime(
            2026,
            9,
            1,
            tzinfo=timezone.utc,
        ),
        confidence=0.90,
        evidence=ExtractionEvidence.INFERRED,
        source_id="import-1",
    )

    memory = candidate.to_memory(
        memory_id="memory-4",
        source_type=SourceType.IMPORT,
    )

    assert memory.source_type == SourceType.IMPORT
    assert memory.source_id == "import-1"