from datetime import datetime, timezone

import pytest

from app.memory.models import (
    EvidenceType,
    Memory,
    MemoryStatus,
    MemoryType,
    SourceType,
)
from app.retrieval.temporal_filter import (
    TemporalCandidateFilter,
)


def make_memory(
    memory_id: str,
    valid_from: str,
    valid_to: str | None = None,
    status: MemoryStatus = MemoryStatus.ACTIVE,
) -> Memory:
    return Memory(
        memory_id=memory_id,
        memory_key=f"user.language.{memory_id}",
        subject="user",
        attribute="programming_language",
        value=memory_id,
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
def memories() -> list[Memory]:
    return [
        make_memory(
            "node",
            "2026-01-01T00:00:00Z",
            "2026-06-01T00:00:00Z",
        ),
        make_memory(
            "go",
            "2026-06-01T00:00:00Z",
            "2026-09-01T00:00:00Z",
        ),
        make_memory(
            "python",
            "2026-09-01T00:00:00Z",
        ),
    ]


@pytest.fixture
def temporal_filter() -> TemporalCandidateFilter:
    return TemporalCandidateFilter()


def test_filter_at_time_returns_memory_valid_at_time(
    temporal_filter,
    memories,
):
    results = temporal_filter.filter_at_time(
        memories,
        datetime(
            2026,
            3,
            15,
            tzinfo=timezone.utc,
        ),
    )

    assert [
        memory.memory_id
        for memory in results
    ] == ["node"]


def test_filter_at_time_excludes_memory_at_valid_to_boundary(
    temporal_filter,
    memories,
):
    results = temporal_filter.filter_at_time(
        memories,
        datetime(
            2026,
            6,
            1,
            tzinfo=timezone.utc,
        ),
    )

    assert [
        memory.memory_id
        for memory in results
    ] == ["go"]


def test_filter_at_time_includes_open_ended_memory(
    temporal_filter,
    memories,
):
    results = temporal_filter.filter_at_time(
        memories,
        datetime(
            2026,
            12,
            1,
            tzinfo=timezone.utc,
        ),
    )

    assert [
        memory.memory_id
        for memory in results
    ] == ["python"]


def test_filter_at_time_excludes_future_memory(
    temporal_filter,
    memories,
):
    results = temporal_filter.filter_at_time(
        memories,
        datetime(
            2025,
            12,
            1,
            tzinfo=timezone.utc,
        ),
    )

    assert results == []


def test_filter_at_time_excludes_discarded_memory(
    temporal_filter,
):
    memories = [
        make_memory(
            "discarded",
            "2026-01-01T00:00:00Z",
            status=MemoryStatus.DISCARDED,
        )
    ]

    results = temporal_filter.filter_at_time(
        memories,
        datetime(
            2026,
            3,
            1,
            tzinfo=timezone.utc,
        ),
    )

    assert results == []


def test_filter_at_time_rejects_naive_datetime(
    temporal_filter,
    memories,
):
    with pytest.raises(
        ValueError,
        match="at must be timezone-aware",
    ):
        temporal_filter.filter_at_time(
            memories,
            datetime(2026, 3, 1),
        )


def test_filter_range_returns_overlapping_memories(
    temporal_filter,
    memories,
):
    results = temporal_filter.filter_range(
        memories,
        start=datetime(
            2026,
            5,
            1,
            tzinfo=timezone.utc,
        ),
        end=datetime(
            2026,
            7,
            1,
            tzinfo=timezone.utc,
        ),
    )

    assert [
        memory.memory_id
        for memory in results
    ] == ["node", "go"]


def test_filter_range_excludes_memory_ending_at_start(
    temporal_filter,
):
    memory = make_memory(
        "node",
        "2026-01-01T00:00:00Z",
        "2026-06-01T00:00:00Z",
    )

    results = temporal_filter.filter_range(
        [memory],
        start=datetime(
            2026,
            6,
            1,
            tzinfo=timezone.utc,
        ),
        end=datetime(
            2026,
            7,
            1,
            tzinfo=timezone.utc,
        ),
    )

    assert results == []


def test_filter_range_excludes_memory_starting_at_end(
    temporal_filter,
):
    memory = make_memory(
        "python",
        "2026-09-01T00:00:00Z",
    )

    results = temporal_filter.filter_range(
        [memory],
        start=datetime(
            2026,
            7,
            1,
            tzinfo=timezone.utc,
        ),
        end=datetime(
            2026,
            9,
            1,
            tzinfo=timezone.utc,
        ),
    )

    assert results == []


def test_filter_range_includes_open_ended_memory(
    temporal_filter,
):
    memory = make_memory(
        "python",
        "2026-09-01T00:00:00Z",
    )

    results = temporal_filter.filter_range(
        [memory],
        start=datetime(
            2026,
            10,
            1,
            tzinfo=timezone.utc,
        ),
        end=datetime(
            2026,
            11,
            1,
            tzinfo=timezone.utc,
        ),
    )

    assert [
        memory.memory_id
        for memory in results
    ] == ["python"]


def test_filter_range_rejects_naive_start(
    temporal_filter,
    memories,
):
    with pytest.raises(
        ValueError,
        match="start must be timezone-aware",
    ):
        temporal_filter.filter_range(
            memories,
            start=datetime(2026, 1, 1),
            end=datetime(
                2026,
                2,
                1,
                tzinfo=timezone.utc,
            ),
        )


def test_filter_range_rejects_naive_end(
    temporal_filter,
    memories,
):
    with pytest.raises(
        ValueError,
        match="end must be timezone-aware",
    ):
        temporal_filter.filter_range(
            memories,
            start=datetime(
                2026,
                1,
                1,
                tzinfo=timezone.utc,
            ),
            end=datetime(2026, 2, 1),
        )


def test_filter_range_rejects_invalid_interval(
    temporal_filter,
    memories,
):
    with pytest.raises(
        ValueError,
        match="end must be later than start",
    ):
        temporal_filter.filter_range(
            memories,
            start=datetime(
                2026,
                3,
                1,
                tzinfo=timezone.utc,
            ),
            end=datetime(
                2026,
                3,
                1,
                tzinfo=timezone.utc,
            ),
        )