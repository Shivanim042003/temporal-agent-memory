import pytest

from app.retrieval.faiss_index import FAISSIndex
from app.retrieval.semantic_retriever import (
    SearchResult,
    SemanticRetriever,
)


def test_search_returns_search_results():
    index = FAISSIndex(3)

    index.add([1.0, 0.0, 0.0])
    index.add([0.0, 1.0, 0.0])

    retriever = SemanticRetriever(index)

    results = retriever.search(
        [1.0, 0.0, 0.0],
        k=2,
    )

    assert len(results) == 2

    assert isinstance(
        results[0],
        SearchResult,
    )

    assert results[0].vector_id == 0
    assert results[0].score == pytest.approx(1.0)


def test_search_results_are_ordered_by_faiss_score():
    index = FAISSIndex(3)

    index.add([1.0, 0.0, 0.0])
    index.add([0.8, 0.2, 0.0])
    index.add([0.0, 1.0, 0.0])

    retriever = SemanticRetriever(index)

    results = retriever.search(
        [1.0, 0.0, 0.0],
        k=3,
    )

    assert [
        result.vector_id
        for result in results
    ] == [0, 1, 2]

    assert (
        results[0].score
        > results[1].score
        > results[2].score
    )


def test_search_empty_index_returns_no_results():
    index = FAISSIndex(3)

    retriever = SemanticRetriever(index)

    results = retriever.search(
        [1.0, 0.0, 0.0],
        k=5,
    )

    assert results == []