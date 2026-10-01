from datetime import datetime, timezone

import pytest

from app.memory.models import (
    EvidenceType,
    Memory,
    MemoryStatus,
    MemoryType,
    SourceType,
)
from app.retrieval.conflict_detector import (
    ConflictDetector,
)


def make_memory(
    memory_id: str,
    value: str,
    memory_key: str = "user.programming_language",
    valid_from: str = "2026-01-01T00:00:00Z",
    valid_to: str | None = None,
    status: MemoryStatus = MemoryStatus.ACTIVE,
) -> Memory:
    return Memory(
        memory_id=memory_id,
        memory_key=memory_key,
        subject="user",
        attribute="programming_language",
        value=value,
        memory_type=MemoryType.SKILL,
        valid_from=valid_from,
        valid_to=valid_to,
        source_type=SourceType.CONVERSATION,
        source_id="conversation_1",
        evidence_type=EvidenceType.EXPLICIT,
        confidence=1.0,
        status=status,
    )


@pytest.fixture
def detector() -> ConflictDetector:
    return ConflictDetector()


def test_detects_overlapping_conflicting_memories(
    detector,
):
    memories = [
        make_memory(
            "python",
            "Python",
            valid_from="2026-01-01T00:00:00Z",
            valid_to="2026-06-01T00:00:00Z",
        ),
        make_memory(
            "java",
            "Java",
            valid_from="2026-03-01T00:00:00Z",
            valid_to="2026-09-01T00:00:00Z",
        ),
    ]

    conflicts = detector.detect(memories)

    assert len(conflicts) == 1
    assert conflicts[0].memory_key == (
        "user.programming_language"
    )
    assert conflicts[0].memory_ids == (
        "java",
        "python",
    )


def test_same_value_is_not_a_conflict(
    detector,
):
    memories = [
        make_memory(
            "python_a",
            "Python",
            valid_from="2026-01-01T00:00:00Z",
            valid_to="2026-06-01T00:00:00Z",
        ),
        make_memory(
            "python_b",
            "Python",
            valid_from="2026-03-01T00:00:00Z",
            valid_to="2026-09-01T00:00:00Z",
        ),
    ]

    conflicts = detector.detect(memories)

    assert conflicts == []


def test_non_overlapping_memories_are_not_conflict(
    detector,
):
    memories = [
        make_memory(
            "python",
            "Python",
            valid_from="2026-01-01T00:00:00Z",
            valid_to="2026-06-01T00:00:00Z",
        ),
        make_memory(
            "java",
            "Java",
            valid_from="2026-06-01T00:00:00Z",
            valid_to="2026-09-01T00:00:00Z",
        ),
    ]

    conflicts = detector.detect(memories)

    assert conflicts == []


def test_open_ended_memory_conflicts_with_later_memory(
    detector,
):
    memories = [
        make_memory(
            "python",
            "Python",
            valid_from="2026-01-01T00:00:00Z",
        ),
        make_memory(
            "java",
            "Java",
            valid_from="2026-06-01T00:00:00Z",
        ),
    ]

    conflicts = detector.detect(memories)

    assert len(conflicts) == 1
    assert conflicts[0].memory_ids == (
        "java",
        "python",
    )


def test_different_memory_keys_do_not_conflict(
    detector,
):
    memories = [
        make_memory(
            "python",
            "Python",
            memory_key="user.programming_language",
        ),
        make_memory(
            "student",
            "Student",
            memory_key="user.occupation",
        ),
    ]

    conflicts = detector.detect(memories)

    assert conflicts == []


def test_discarded_memory_is_ignored(
    detector,
):
    memories = [
        make_memory(
            "python",
            "Python",
            valid_from="2026-01-01T00:00:00Z",
            valid_to="2026-06-01T00:00:00Z",
            status=MemoryStatus.DISCARDED,
        ),
        make_memory(
            "java",
            "Java",
            valid_from="2026-03-01T00:00:00Z",
            valid_to="2026-09-01T00:00:00Z",
        ),
    ]

    conflicts = detector.detect(memories)

    assert conflicts == []


def test_empty_input_returns_no_conflicts(
    detector,
):
    assert detector.detect([]) == []


def test_single_memory_returns_no_conflict(
    detector,
):
    memories = [
        make_memory(
            "python",
            "Python",
        )
    ]

    assert detector.detect(memories) == []


def test_multiple_conflicting_memories_form_one_group(
    detector,
):
    memories = [
        make_memory(
            "python",
            "Python",
            valid_from="2026-01-01T00:00:00Z",
            valid_to="2026-12-01T00:00:00Z",
        ),
        make_memory(
            "java",
            "Java",
            valid_from="2026-03-01T00:00:00Z",
            valid_to="2026-09-01T00:00:00Z",
        ),
        make_memory(
            "go",
            "Go",
            valid_from="2026-04-01T00:00:00Z",
            valid_to="2026-08-01T00:00:00Z",
        ),
    ]

    conflicts = detector.detect(memories)

    assert len(conflicts) == 1
    assert conflicts[0].memory_ids == (
        "go",
        "java",
        "python",
    )


def test_conflicting_memory_keys_create_separate_groups(
    detector,
):
    memories = [
        make_memory(
            "python",
            "Python",
            memory_key="user.programming_language",
        ),
        make_memory(
            "java",
            "Java",
            memory_key="user.programming_language",
        ),
        make_memory(
            "student",
            "Student",
            memory_key="user.occupation",
        ),
        make_memory(
            "engineer",
            "Engineer",
            memory_key="user.occupation",
        ),
    ]

    conflicts = detector.detect(memories)

    assert len(conflicts) == 2

    assert {
        conflict.memory_key
        for conflict in conflicts
    } == {
        "user.programming_language",
        "user.occupation",
    }


def test_touching_intervals_are_not_conflicting(
    detector,
):
    memories = [
        make_memory(
            "python",
            "Python",
            valid_from="2026-01-01T00:00:00Z",
            valid_to="2026-06-01T00:00:00Z",
        ),
        make_memory(
            "java",
            "Java",
            valid_from="2026-06-01T00:00:00Z",
            valid_to="2026-12-01T00:00:00Z",
        ),
    ]

    assert detector.detect(memories) == []