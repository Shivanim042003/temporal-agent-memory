from datetime import datetime

from app.memory.models import Memory, MemoryStatus


class TemporalCandidateFilter:
    def filter_at_time(
        self,
        memories: list[Memory],
        at: datetime,
    ) -> list[Memory]:
        if at.tzinfo is None:
            raise ValueError(
                "at must be timezone-aware"
            )

        results: list[Memory] = []

        for memory in memories:
            if memory.status == MemoryStatus.DISCARDED:
                continue

            if memory.valid_from > at:
                continue

            if (
                memory.valid_to is not None
                and at >= memory.valid_to
            ):
                continue

            results.append(memory)

        return results

    def filter_range(
        self,
        memories: list[Memory],
        start: datetime,
        end: datetime,
    ) -> list[Memory]:
        if start.tzinfo is None:
            raise ValueError(
                "start must be timezone-aware"
            )

        if end.tzinfo is None:
            raise ValueError(
                "end must be timezone-aware"
            )

        if end <= start:
            raise ValueError(
                "end must be later than start"
            )

        results: list[Memory] = []

        for memory in memories:
            if memory.status == MemoryStatus.DISCARDED:
                continue

            memory_end = memory.valid_to

            if memory_end is not None and memory_end <= start:
                continue

            if memory.valid_from >= end:
                continue

            results.append(memory)

        return results