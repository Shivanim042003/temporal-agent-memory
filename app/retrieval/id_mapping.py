import json
from pathlib import Path


class MemoryIDMapping:
    def __init__(self):
        self._vector_to_memory: dict[int, str] = {}
        self._memory_to_vector: dict[str, int] = {}

    @property
    def size(self) -> int:
        return len(self._vector_to_memory)

    def add(
        self,
        vector_id: int,
        memory_id: str,
    ) -> None:
        if vector_id < 0:
            raise ValueError(
                "vector_id must be non-negative"
            )

        if not memory_id:
            raise ValueError(
                "memory_id must not be empty"
            )

        if vector_id in self._vector_to_memory:
            raise ValueError(
                "vector_id already exists"
            )

        if memory_id in self._memory_to_vector:
            raise ValueError(
                "memory_id already exists"
            )

        self._vector_to_memory[vector_id] = memory_id
        self._memory_to_vector[memory_id] = vector_id

    def get_memory_id(
        self,
        vector_id: int,
    ) -> str | None:
        return self._vector_to_memory.get(vector_id)

    def get_vector_id(
        self,
        memory_id: str,
    ) -> int | None:
        return self._memory_to_vector.get(memory_id)

    def remove_by_vector_id(
        self,
        vector_id: int,
    ) -> str | None:
        memory_id = self._vector_to_memory.pop(
            vector_id,
            None,
        )

        if memory_id is None:
            return None

        self._memory_to_vector.pop(
            memory_id,
            None,
        )

        return memory_id

    def remove_by_memory_id(
        self,
        memory_id: str,
    ) -> int | None:
        vector_id = self._memory_to_vector.pop(
            memory_id,
            None,
        )

        if vector_id is None:
            return None

        self._vector_to_memory.pop(
            vector_id,
            None,
        )

        return vector_id

    def save(
        self,
        path: Path | str,
    ) -> None:
        path = Path(path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        data = {
            str(vector_id): memory_id
            for vector_id, memory_id
            in self._vector_to_memory.items()
        }

        path.write_text(
            json.dumps(
                data,
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(
        cls,
        path: Path | str,
    ) -> "MemoryIDMapping":
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(
                f"memory ID mapping does not exist: {path}"
            )

        data = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

        if not isinstance(data, dict):
            raise ValueError(
                "memory ID mapping must contain a JSON object"
            )

        mapping = cls()

        for vector_id, memory_id in data.items():
            if not isinstance(vector_id, str):
                raise ValueError(
                    "vector IDs must be stored as strings"
                )

            if not isinstance(memory_id, str):
                raise ValueError(
                    "memory IDs must be strings"
                )

            mapping.add(
                vector_id=int(vector_id),
                memory_id=memory_id,
            )

        return mapping