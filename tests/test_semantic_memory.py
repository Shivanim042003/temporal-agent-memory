from pathlib import Path
from unittest.mock import MagicMock

import pytest

from app.memory.models import (
    EvidenceType,
    Memory,
    MemoryType,
    SourceType,
)
from app.retrieval.faiss_index import FAISSIndex
from app.retrieval.id_mapping import MemoryIDMapping
from app.retrieval.semantic_memory import (
    SemanticMemoryIndex,
    SemanticMemoryResult,
)
from app.storage.database import initialize_database
from app.storage.memory_repository import insert


def make_memory(
    memory_id: str = "mem_1",
    value: str = "Python",
) -> Memory:
    return Memory(
        memory_id=memory_id,
        memory_key="user.programming_language",
        subject="user",
        attribute="programming_language",
        value=value,
        memory_type=MemoryType.SKILL,
        valid_from="2026-01-01T00:00:00Z",
        source_type=SourceType.CONVERSATION,
        source_id="conversation_1",
        evidence_type=EvidenceType.EXPLICIT,
        confidence=1.0,
    )


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    path = tmp_path / "memory.db"

    initialize_database(path)

    return path


@pytest.fixture
def embedding_model():
    model = MagicMock()

    model.embed_memory.side_effect = (
        lambda memory: {
            "mem_1": [1.0, 0.0, 0.0],
            "mem_2": [0.0, 1.0, 0.0],
        }[memory.memory_id]
    )

    model.embed_text.return_value = [
        1.0,
        0.0,
        0.0,
    ]

    return model


@pytest.fixture
def semantic_memory(embedding_model):
    return SemanticMemoryIndex(
        embedding_model=embedding_model,
        faiss_index=FAISSIndex(3),
        id_mapping=MemoryIDMapping(),
    )


def test_index_starts_empty(
    semantic_memory,
):
    assert semantic_memory.size == 0


def test_add_embeds_memory_and_adds_to_faiss(
    semantic_memory,
    embedding_model,
):
    memory = make_memory()

    vector_id = semantic_memory.add(memory)

    assert vector_id == 0
    assert semantic_memory.size == 1

    embedding_model.embed_memory.assert_called_once_with(
        memory
    )


def test_add_creates_memory_id_mapping(
    semantic_memory,
):
    memory = make_memory()

    vector_id = semantic_memory.add(memory)

    assert (
        semantic_memory.id_mapping.get_memory_id(
            vector_id
        )
        == "mem_1"
    )


def test_search_returns_memory_from_sqlite(
    semantic_memory,
    embedding_model,
    db_path,
):
    memory = make_memory()

    insert(
        memory,
        db_path=db_path,
    )

    semantic_memory.add(memory)

    results = semantic_memory.search(
        query="What language am I using?",
        k=1,
        db_path=db_path,
    )

    assert len(results) == 1

    result = results[0]

    assert isinstance(
        result,
        SemanticMemoryResult,
    )

    assert result.memory.memory_id == "mem_1"
    assert result.memory.value == "Python"
    assert result.score == pytest.approx(1.0)

    embedding_model.embed_text.assert_called_once_with(
        "What language am I using?"
    )


def test_search_returns_results_in_faiss_order(
    semantic_memory,
    embedding_model,
    db_path,
):
    memory_1 = make_memory(
        memory_id="mem_1",
        value="Python",
    )

    memory_2 = make_memory(
        memory_id="mem_2",
        value="Java",
    )

    insert(
        memory_1,
        db_path=db_path,
    )

    insert(
        memory_2,
        db_path=db_path,
    )

    semantic_memory.add(memory_1)
    semantic_memory.add(memory_2)

    results = semantic_memory.search(
        query="programming language",
        k=2,
        db_path=db_path,
    )

    assert len(results) == 2

    assert [
        result.memory.memory_id
        for result in results
    ] == [
        "mem_1",
        "mem_2",
    ]

    assert results[0].score > results[1].score


def test_search_rejects_empty_query(
    semantic_memory,
):
    with pytest.raises(
        ValueError,
        match="query must not be empty",
    ):
        semantic_memory.search(
            query="   ",
            k=1,
        )


def test_search_skips_missing_memory(
    semantic_memory,
    db_path,
):
    memory = make_memory()

    semantic_memory.add(memory)

    results = semantic_memory.search(
        query="programming language",
        k=1,
        db_path=db_path,
    )

    assert results == []


def test_save_persists_faiss_and_mapping(
    semantic_memory,
    tmp_path,
):
    memory = make_memory()

    semantic_memory.add(memory)

    index_path = (
        tmp_path / "indexes" / "memory.faiss"
    )

    mapping_path = (
        tmp_path / "indexes" / "memory_ids.json"
    )

    semantic_memory.save(
        index_path=index_path,
        mapping_path=mapping_path,
    )

    assert index_path.exists()
    assert mapping_path.exists()


def test_load_restores_semantic_memory_index(
    semantic_memory,
    embedding_model,
    tmp_path,
):
    memory = make_memory()

    semantic_memory.add(memory)

    index_path = (
        tmp_path / "memory.faiss"
    )

    mapping_path = (
        tmp_path / "memory_ids.json"
    )

    semantic_memory.save(
        index_path=index_path,
        mapping_path=mapping_path,
    )

    loaded = SemanticMemoryIndex.load(
        embedding_model=embedding_model,
        index_path=index_path,
        mapping_path=mapping_path,
    )

    assert loaded.size == 1

    assert (
        loaded.id_mapping.get_memory_id(0)
        == "mem_1"
    )


def test_load_rejects_mismatched_index_and_mapping(
    semantic_memory,
    tmp_path,
):
    memory = make_memory()

    semantic_memory.add(memory)

    index_path = (
        tmp_path / "memory.faiss"
    )

    mapping_path = (
        tmp_path / "memory_ids.json"
    )

    semantic_memory.faiss_index.save(
        index_path
    )

    empty_mapping = MemoryIDMapping()

    empty_mapping.save(
        mapping_path
    )

    with pytest.raises(
        ValueError,
        match=(
            "FAISS index size does not match "
            "memory ID mapping size"
        ),
    ):
        SemanticMemoryIndex.load(
            embedding_model=semantic_memory.embedding_model,
            index_path=index_path,
            mapping_path=mapping_path,
        )