from dataclasses import dataclass

from app.memory.models import Memory, MemoryStatus


@dataclass(frozen=True)
class ConflictGroup:
    memory_key: str
    memory_ids: tuple[str, ...]


class ConflictDetector:
    def detect(
        self,
        memories: list[Memory],
    ) -> list[ConflictGroup]:
        eligible_memories = [
            memory
            for memory in memories
            if memory.status != MemoryStatus.DISCARDED
        ]

        groups: dict[str, list[Memory]] = {}

        for memory in eligible_memories:
            groups.setdefault(
                memory.memory_key,
                [],
            ).append(memory)

        conflicts: list[ConflictGroup] = []

        for memory_key, candidates in groups.items():
            conflict_ids = self._find_conflicting_ids(
                candidates
            )

            if len(conflict_ids) >= 2:
                conflicts.append(
                    ConflictGroup(
                        memory_key=memory_key,
                        memory_ids=tuple(conflict_ids),
                    )
                )

        return conflicts

    def _find_conflicting_ids(
        self,
        memories: list[Memory],
    ) -> list[str]:
        conflicting_ids: set[str] = set()

        for index, first in enumerate(memories):
            for second in memories[index + 1:]:
                if first.value == second.value:
                    continue

                if not self._intervals_overlap(
                    first,
                    second,
                ):
                    continue

                conflicting_ids.add(
                    first.memory_id
                )
                conflicting_ids.add(
                    second.memory_id
                )

        return sorted(conflicting_ids)

    @staticmethod
    def _intervals_overlap(
        first: Memory,
        second: Memory,
    ) -> bool:
        if (
            first.valid_to is not None
            and first.valid_to <= second.valid_from
        ):
            return False

        if (
            second.valid_to is not None
            and second.valid_to <= first.valid_from
        ):
            return False

        return True