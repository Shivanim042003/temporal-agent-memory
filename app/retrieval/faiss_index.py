from pathlib import Path

import faiss
import numpy as np


class FAISSIndex:
    def __init__(self, dimension: int):
        if dimension <= 0:
            raise ValueError(
                "dimension must be greater than zero"
            )

        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)

    @property
    def size(self) -> int:
        return self.index.ntotal

    def add(self, embedding: list[float]) -> int:
        vector = self._prepare_vector(embedding)

        self.index.add(vector)

        return self.index.ntotal - 1

    def search(
        self,
        embedding: list[float],
        k: int,
    ) -> tuple[list[float], list[int]]:
        if k <= 0:
            raise ValueError(
                "k must be greater than zero"
            )

        if self.size == 0:
            return [], []

        vector = self._prepare_vector(embedding)

        actual_k = min(k, self.size)

        scores, ids = self.index.search(
            vector,
            actual_k,
        )

        return (
            scores[0].tolist(),
            ids[0].tolist(),
        )

    def save(
        self,
        path: Path | str,
    ) -> None:
        path = Path(path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        faiss.write_index(
            self.index,
            str(path),
        )

    @classmethod
    def load(
        cls,
        path: Path | str,
    ) -> "FAISSIndex":
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(
                f"FAISS index does not exist: {path}"
            )

        index = faiss.read_index(
            str(path),
        )

        instance = cls(
            dimension=index.d
        )

        instance.index = index

        return instance

    def _prepare_vector(
        self,
        embedding: list[float],
    ) -> np.ndarray:
        if not embedding:
            raise ValueError(
                "embedding must not be empty"
            )

        vector = np.asarray(
            embedding,
            dtype=np.float32,
        )

        if vector.ndim != 1:
            raise ValueError(
                "embedding must be one-dimensional"
            )

        if vector.shape[0] != self.dimension:
            raise ValueError(
                "embedding dimension does not match index dimension"
            )

        if not np.isfinite(vector).all():
            raise ValueError(
                "embedding must contain only finite values"
            )

        return vector.reshape(1, -1)