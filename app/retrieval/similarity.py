import numpy as np


def cosine_similarity(
    embedding_a: list[float],
    embedding_b: list[float],
) -> float:
    if not embedding_a:
        raise ValueError("embedding_a must not be empty")

    if not embedding_b:
        raise ValueError("embedding_b must not be empty")

    vector_a = np.asarray(embedding_a, dtype=np.float32)
    vector_b = np.asarray(embedding_b, dtype=np.float32)

    if vector_a.ndim != 1:
        raise ValueError("embedding_a must be one-dimensional")

    if vector_b.ndim != 1:
        raise ValueError("embedding_b must be one-dimensional")

    if vector_a.shape != vector_b.shape:
        raise ValueError(
            "embeddings must have the same dimension"
        )

    if not np.isfinite(vector_a).all():
        raise ValueError(
            "embedding_a must contain only finite values"
        )

    if not np.isfinite(vector_b).all():
        raise ValueError(
            "embedding_b must contain only finite values"
        )

    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)

    if norm_a == 0:
        raise ValueError("embedding_a must not be a zero vector")

    if norm_b == 0:
        raise ValueError("embedding_b must not be a zero vector")

    return float(
        np.dot(vector_a, vector_b) / (norm_a * norm_b)
    )