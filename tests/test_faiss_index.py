import pytest

from app.retrieval.faiss_index import FAISSIndex


def test_index_starts_empty():
    index = FAISSIndex(3)

    assert index.dimension == 3
    assert index.size == 0


def test_add_embedding_increases_index_size():
    index = FAISSIndex(3)

    vector_id = index.add(
        [1.0, 0.0, 0.0]
    )

    assert vector_id == 0
    assert index.size == 1


def test_add_multiple_embeddings_returns_sequential_ids():
    index = FAISSIndex(3)

    first_id = index.add(
        [1.0, 0.0, 0.0]
    )

    second_id = index.add(
        [0.0, 1.0, 0.0]
    )

    third_id = index.add(
        [0.0, 0.0, 1.0]
    )

    assert first_id == 0
    assert second_id == 1
    assert third_id == 2
    assert index.size == 3


def test_search_returns_most_similar_vector():
    index = FAISSIndex(3)

    index.add([1.0, 0.0, 0.0])
    index.add([0.0, 1.0, 0.0])
    index.add([0.0, 0.0, 1.0])

    scores, ids = index.search(
        [1.0, 0.0, 0.0],
        k=1,
    )

    assert ids == [0]
    assert scores[0] == pytest.approx(1.0)


def test_search_returns_top_k_vectors():
    index = FAISSIndex(3)

    index.add([1.0, 0.0, 0.0])
    index.add([0.9, 0.1, 0.0])
    index.add([0.0, 1.0, 0.0])

    scores, ids = index.search(
        [1.0, 0.0, 0.0],
        k=2,
    )

    assert ids == [0, 1]
    assert scores[0] > scores[1]


def test_search_limits_k_to_index_size():
    index = FAISSIndex(3)

    index.add([1.0, 0.0, 0.0])

    scores, ids = index.search(
        [1.0, 0.0, 0.0],
        k=10,
    )

    assert len(scores) == 1
    assert len(ids) == 1
    assert ids == [0]


def test_search_empty_index_returns_empty_results():
    index = FAISSIndex(3)

    scores, ids = index.search(
        [1.0, 0.0, 0.0],
        k=3,
    )

    assert scores == []
    assert ids == []


def test_rejects_invalid_dimension():
    with pytest.raises(
        ValueError,
        match="dimension must be greater than zero",
    ):
        FAISSIndex(0)


def test_add_rejects_empty_embedding():
    index = FAISSIndex(3)

    with pytest.raises(
        ValueError,
        match="embedding must not be empty",
    ):
        index.add([])


def test_search_rejects_invalid_k():
    index = FAISSIndex(3)

    with pytest.raises(
        ValueError,
        match="k must be greater than zero",
    ):
        index.search(
            [1.0, 0.0, 0.0],
            k=0,
        )


def test_rejects_dimension_mismatch():
    index = FAISSIndex(3)

    with pytest.raises(
        ValueError,
        match="embedding dimension does not match index dimension",
    ):
        index.add(
            [1.0, 0.0]
        )


def test_rejects_non_finite_embedding():
    index = FAISSIndex(3)

    with pytest.raises(
        ValueError,
        match="embedding must contain only finite values",
    ):
        index.add(
            [1.0, float("nan"), 0.0]
        )

def test_save_creates_faiss_file(tmp_path):
    index = FAISSIndex(3)

    index.add([1.0, 0.0, 0.0])

    path = tmp_path / "memory.faiss"

    index.save(path)

    assert path.exists()
    assert path.is_file()


def test_load_restores_index(tmp_path):
    index = FAISSIndex(3)

    index.add([1.0, 0.0, 0.0])
    index.add([0.0, 1.0, 0.0])

    path = tmp_path / "memory.faiss"

    index.save(path)

    loaded = FAISSIndex.load(path)

    assert loaded.dimension == 3
    assert loaded.size == 2


def test_loaded_index_can_search(tmp_path):
    index = FAISSIndex(3)

    index.add([1.0, 0.0, 0.0])
    index.add([0.0, 1.0, 0.0])

    path = tmp_path / "memory.faiss"

    index.save(path)

    loaded = FAISSIndex.load(path)

    scores, ids = loaded.search(
        [1.0, 0.0, 0.0],
        k=2,
    )

    assert ids == [0, 1]
    assert scores[0] == pytest.approx(1.0)


def test_save_creates_parent_directories(tmp_path):
    index = FAISSIndex(3)

    index.add([1.0, 0.0, 0.0])

    path = (
        tmp_path
        / "indexes"
        / "nested"
        / "memory.faiss"
    )

    index.save(path)

    assert path.exists()


def test_load_rejects_missing_file(tmp_path):
    path = tmp_path / "does_not_exist.faiss"

    with pytest.raises(
        FileNotFoundError,
        match="FAISS index does not exist",
    ):
        FAISSIndex.load(path)        