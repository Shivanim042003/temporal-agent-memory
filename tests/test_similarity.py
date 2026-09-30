import pytest

from app.retrieval.similarity import cosine_similarity


def test_identical_vectors_have_similarity_one():
    result = cosine_similarity(
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    )

    assert result == pytest.approx(1.0)


def test_orthogonal_vectors_have_similarity_zero():
    result = cosine_similarity(
        [1.0, 0.0],
        [0.0, 1.0],
    )

    assert result == pytest.approx(0.0)


def test_opposite_vectors_have_similarity_negative_one():
    result = cosine_similarity(
        [1.0, 0.0],
        [-1.0, 0.0],
    )

    assert result == pytest.approx(-1.0)


def test_similarity_is_symmetric():
    embedding_a = [0.2, 0.4, 0.6]
    embedding_b = [0.1, 0.3, 0.5]

    first = cosine_similarity(
        embedding_a,
        embedding_b,
    )

    second = cosine_similarity(
        embedding_b,
        embedding_a,
    )

    assert first == pytest.approx(second)


def test_similarity_handles_non_normalized_vectors():
    result = cosine_similarity(
        [2.0, 0.0],
        [4.0, 0.0],
    )

    assert result == pytest.approx(1.0)


def test_similarity_rejects_empty_first_embedding():
    with pytest.raises(
        ValueError,
        match="embedding_a must not be empty",
    ):
        cosine_similarity(
            [],
            [1.0],
        )


def test_similarity_rejects_empty_second_embedding():
    with pytest.raises(
        ValueError,
        match="embedding_b must not be empty",
    ):
        cosine_similarity(
            [1.0],
            [],
        )


def test_similarity_rejects_dimension_mismatch():
    with pytest.raises(
        ValueError,
        match="same dimension",
    ):
        cosine_similarity(
            [1.0, 0.0],
            [1.0],
        )


def test_similarity_rejects_zero_first_vector():
    with pytest.raises(
        ValueError,
        match="embedding_a must not be a zero vector",
    ):
        cosine_similarity(
            [0.0, 0.0],
            [1.0, 0.0],
        )


def test_similarity_rejects_zero_second_vector():
    with pytest.raises(
        ValueError,
        match="embedding_b must not be a zero vector",
    ):
        cosine_similarity(
            [1.0, 0.0],
            [0.0, 0.0],
        )


def test_similarity_rejects_non_finite_first_embedding():
    with pytest.raises(
        ValueError,
        match="embedding_a must contain only finite values",
    ):
        cosine_similarity(
            [1.0, float("nan")],
            [1.0, 0.0],
        )


def test_similarity_rejects_non_finite_second_embedding():
    with pytest.raises(
        ValueError,
        match="embedding_b must contain only finite values",
    ):
        cosine_similarity(
            [1.0, 0.0],
            [1.0, float("inf")],
        )