from datetime import datetime, timezone, timedelta

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


def test_valid_memory_constructs():
    memory = Memory(**make_memory())

    assert memory.memory_id == "mem_001"
    assert memory.memory_key == "primary_backend_language"
    assert memory.value == "Node.js"
    assert memory.status == MemoryStatus.ACTIVE


def test_valid_to_must_be_after_valid_from():
    with pytest.raises(ValidationError):
        Memory(
            **make_memory(
                valid_from=datetime(2026, 6, 1, tzinfo=timezone.utc),
                valid_to=datetime(2026, 1, 1, tzinfo=timezone.utc),
            )
        )


def test_naive_datetime_is_rejected():
    with pytest.raises(ValidationError):
        Memory(
            **make_memory(
                valid_from=datetime(2026, 1, 1),
            )
        )


def test_datetime_with_plus_0530_offset_is_normalized_to_utc():
    ist = timezone(timedelta(hours=5, minutes=30))

    memory = Memory(
        **make_memory(
            valid_from=datetime(2026, 1, 1, 5, 30, tzinfo=ist),
        )
    )

    assert memory.valid_from == datetime(
        2026, 1, 1, 0, 0, tzinfo=timezone.utc
    )
    assert memory.valid_from.tzinfo == timezone.utc


def test_confidence_above_one_is_rejected():
    with pytest.raises(ValidationError):
        Memory(**make_memory(confidence=1.5))


def test_empty_memory_key_is_rejected():
    with pytest.raises(ValidationError):
        Memory(**make_memory(memory_key=""))


def test_past_valid_to_with_active_status_is_accepted():
    memory = Memory(
        **make_memory(
            valid_to=datetime(2026, 6, 1, tzinfo=timezone.utc),
            status=MemoryStatus.ACTIVE,
        )
    )

    assert memory.status == MemoryStatus.ACTIVE
    assert memory.valid_to == datetime(
        2026, 6, 1, tzinfo=timezone.utc
    )


def test_superseded_memory_requires_valid_to():
    with pytest.raises(ValidationError):
        Memory(
            **make_memory(
                valid_to=None,
                status=MemoryStatus.SUPERSEDED,
            )
        )


def test_memory_cannot_supersede_itself():
    with pytest.raises(ValidationError):
        Memory(
            **make_memory(
                supersedes_id="mem_001",
            )
        )


def test_memory_is_frozen():
    memory = Memory(**make_memory())

    with pytest.raises(ValidationError):
        memory.value = "Python"


def test_consolidated_memory_requires_canonical_memory_id():
    with pytest.raises(ValidationError):
        Memory(
            **make_memory(
                status=MemoryStatus.CONSOLIDATED,
                canonical_memory_id=None,
            )
        )


def test_canonical_memory_id_requires_consolidated_status():
    with pytest.raises(ValidationError):
        Memory(
            **make_memory(
                status=MemoryStatus.ACTIVE,
                canonical_memory_id="mem_002",
            )
        )


def test_consolidated_memory_with_canonical_memory_id_is_valid():
    memory = Memory(
        **make_memory(
            status=MemoryStatus.CONSOLIDATED,
            canonical_memory_id="mem_002",
        )
    )

    assert memory.status == MemoryStatus.CONSOLIDATED
    assert memory.canonical_memory_id == "mem_002"


def test_consolidated_memory_can_have_open_ended_interval():
    memory = Memory(
        **make_memory(
            valid_to=None,
            status=MemoryStatus.CONSOLIDATED,
            canonical_memory_id="mem_002",
        )
    )

    assert memory.status == MemoryStatus.CONSOLIDATED
    assert memory.valid_to is None
    assert memory.canonical_memory_id == "mem_002"


def test_memory_cannot_be_canonical_for_itself():
    with pytest.raises(ValidationError):
        Memory(
            **make_memory(
                status=MemoryStatus.CONSOLIDATED,
                canonical_memory_id="mem_001",
            )
        )


def test_discarded_memory_preserves_original_interval():
    memory = Memory(
        **make_memory(
            valid_from=datetime(2026, 1, 1, tzinfo=timezone.utc),
            valid_to=datetime(2026, 6, 1, tzinfo=timezone.utc),
            status=MemoryStatus.DISCARDED,
        )
    )

    assert memory.status == MemoryStatus.DISCARDED
    assert memory.valid_from == datetime(
        2026, 1, 1, tzinfo=timezone.utc
    )
    assert memory.valid_to == datetime(
        2026, 6, 1, tzinfo=timezone.utc
    )