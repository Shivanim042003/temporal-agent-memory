from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from app.memory.models import (
    EvidenceType,
    Memory,
    MemoryStatus,
    MemoryType,
    SourceType,
    TimePrecision,
)


def make_memory(**overrides):
    values = {
        "memory_id": "mem_1",
        "memory_key": "user.programming_language",
        "subject": "user",
        "attribute": "programming_language",
        "value": "Python",
        "memory_type": MemoryType.SKILL,
        "valid_from": datetime(2026, 1, 1, tzinfo=timezone.utc),
        "source_type": SourceType.CONVERSATION,
        "source_id": "conversation_1",
        "evidence_type": EvidenceType.EXPLICIT,
        "confidence": 0.95,
    }

    values.update(overrides)

    return Memory(**values)


def test_memory_accepts_valid_data():
    memory = make_memory()

    assert memory.memory_id == "mem_1"
    assert memory.memory_key == "user.programming_language"
    assert memory.subject == "user"
    assert memory.attribute == "programming_language"
    assert memory.value == "Python"
    assert memory.memory_type == MemoryType.SKILL
    assert memory.status == MemoryStatus.ACTIVE


def test_memory_defaults_recorded_at():
    before = datetime.now(timezone.utc)

    memory = make_memory()

    after = datetime.now(timezone.utc)

    assert before <= memory.recorded_at <= after


def test_memory_normalizes_datetime_to_utc():
    memory = make_memory(
        valid_from=datetime(
            2026,
            1,
            1,
            12,
            tzinfo=timezone.utc,
        )
    )

    assert memory.valid_from.tzinfo == timezone.utc


def test_memory_rejects_naive_datetime():
    with pytest.raises(ValidationError):
        make_memory(
            valid_from=datetime(2026, 1, 1)
        )


def test_memory_rejects_invalid_confidence():
    with pytest.raises(ValidationError):
        make_memory(confidence=1.1)

    with pytest.raises(ValidationError):
        make_memory(confidence=-0.1)


def test_memory_rejects_empty_memory_id():
    with pytest.raises(ValidationError):
        make_memory(memory_id="")


def test_memory_rejects_empty_memory_key():
    with pytest.raises(ValidationError):
        make_memory(memory_key="")


def test_memory_rejects_empty_subject():
    with pytest.raises(ValidationError):
        make_memory(subject="")


def test_memory_rejects_empty_attribute():
    with pytest.raises(ValidationError):
        make_memory(attribute="")


def test_memory_rejects_empty_value():
    with pytest.raises(ValidationError):
        make_memory(value="")


def test_memory_rejects_invalid_temporal_interval():
    with pytest.raises(ValidationError):
        make_memory(
            valid_to=datetime(
                2025,
                12,
                31,
                tzinfo=timezone.utc,
            )
        )


def test_historical_memory_requires_valid_to():
    with pytest.raises(ValidationError):
        make_memory(
            status=MemoryStatus.HISTORICAL,
            valid_to=None,
        )


def test_superseded_memory_requires_valid_to():
    with pytest.raises(ValidationError):
        make_memory(
            status=MemoryStatus.SUPERSEDED,
            valid_to=None,
        )


def test_memory_cannot_supersede_itself():
    with pytest.raises(ValidationError):
        make_memory(
            supersedes_id="mem_1",
        )


def test_memory_cannot_be_consolidated_into_itself():
    with pytest.raises(ValidationError):
        make_memory(
            status=MemoryStatus.CONSOLIDATED,
            canonical_memory_id="mem_1",
        )


def test_consolidated_memory_requires_canonical_memory_id():
    with pytest.raises(ValidationError):
        make_memory(
            status=MemoryStatus.CONSOLIDATED,
            canonical_memory_id=None,
        )


def test_non_consolidated_memory_cannot_have_canonical_memory_id():
    with pytest.raises(ValidationError):
        make_memory(
            status=MemoryStatus.ACTIVE,
            canonical_memory_id="canonical_1",
        )


def test_to_embedding_text():
    memory = make_memory(
        subject="user",
        attribute="programming_language",
        value="Python",
    )

    assert (
        memory.to_embedding_text()
        == "The user has programming_language: Python."
    )


def test_to_embedding_text_is_deterministic():
    memory = make_memory()

    first = memory.to_embedding_text()
    second = memory.to_embedding_text()

    assert first == second


def test_to_embedding_text_uses_memory_fields():
    memory = make_memory(
        subject="user",
        attribute="backend_framework",
        value="FastAPI",
    )

    assert (
        memory.to_embedding_text()
        == "The user has backend_framework: FastAPI."
    )

def test_consolidated_memory_can_have_open_ended_interval():
    memory = make_memory(
        status=MemoryStatus.CONSOLIDATED,
        canonical_memory_id="mem_canonical",
        valid_to=None,
    )

    assert memory.status == MemoryStatus.CONSOLIDATED
    assert memory.valid_to is None


def test_discarded_memory_preserves_original_interval():
    memory = make_memory(
        valid_from=datetime(2026, 1, 1, tzinfo=timezone.utc),
        valid_to=datetime(2026, 6, 1, tzinfo=timezone.utc),
        status=MemoryStatus.DISCARDED,
    )

    assert memory.status == MemoryStatus.DISCARDED
    assert memory.valid_from == datetime(2026, 1, 1, tzinfo=timezone.utc)
    assert memory.valid_to == datetime(2026, 6, 1, tzinfo=timezone.utc)


def test_discarded_memory_can_have_open_ended_interval():
    memory = make_memory(
        valid_to=None,
        status=MemoryStatus.DISCARDED,
    )

    assert memory.status == MemoryStatus.DISCARDED
    assert memory.valid_to is None