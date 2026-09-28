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
    data = {
        "memory_id": "mem_001",
        "memory_key": "primary_backend_language",
        "subject": "user",
        "attribute": "uses",
        "value": "Node.js",
        "memory_type": MemoryType.SKILL,
        "valid_from": datetime(2026, 1, 1, tzinfo=timezone.utc),
        "recorded_at": datetime(2026, 1, 15, tzinfo=timezone.utc),
        "precision": TimePrecision.DAY,
        "source_type": SourceType.CONVERSATION,
        "source_id": "conversation_001",
        "evidence_type": EvidenceType.EXPLICIT,
        "confidence": 0.98,
        "status": MemoryStatus.ACTIVE,
    }
    data.update(overrides)
    return data


def test_valid_memory_is_accepted():
    memory = Memory(**make_memory())

    assert memory.memory_id == "mem_001"
    assert memory.value == "Node.js"
    assert memory.valid_from.tzinfo is not None


def test_invalid_interval_is_rejected():
    with pytest.raises(ValidationError):
        Memory(
            **make_memory(
                valid_from=datetime(2026, 6, 1, tzinfo=timezone.utc),
                valid_to=datetime(2026, 1, 1, tzinfo=timezone.utc),
                status=MemoryStatus.HISTORICAL,
            )
        )


def test_naive_datetime_is_rejected():
    with pytest.raises(ValidationError):
        Memory(
            **make_memory(
                valid_from=datetime(2026, 1, 1),
            )
        )


def test_confidence_must_be_between_zero_and_one():
    with pytest.raises(ValidationError):
        Memory(**make_memory(confidence=1.5))


def test_empty_memory_key_is_rejected():
    with pytest.raises(ValidationError):
        Memory(**make_memory(memory_key=""))


def test_active_memory_cannot_end_in_the_past():
    with pytest.raises(ValidationError):
        Memory(
            **make_memory(
                valid_to=datetime(2020, 1, 1, tzinfo=timezone.utc),
                status=MemoryStatus.ACTIVE,
            )
        )


def test_historical_memory_can_end_in_the_past():
    memory = Memory(
        **make_memory(
            valid_to=datetime(2026, 6, 1, tzinfo=timezone.utc),
            status=MemoryStatus.HISTORICAL,
        )
    )

    assert memory.status == MemoryStatus.HISTORICAL
    assert memory.valid_to is not None


def test_memory_is_immutable():
    memory = Memory(**make_memory())

    with pytest.raises(ValidationError):
        memory.value = "Python"
