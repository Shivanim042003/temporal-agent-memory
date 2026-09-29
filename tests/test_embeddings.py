from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.memory.models import Memory
from app.retrieval.embeddings import EmbeddingModel, MODEL_NAME


def make_memory() -> Memory:
    return Memory(
        memory_id="mem_1",
        memory_key="user.programming_language",
        subject="user",
        attribute="programming_language",
        value="Python",
        memory_type="skill",
        valid_from="2026-01-01T00:00:00Z",
        source_type="conversation",
        source_id="conversation_1",
        evidence_type="explicit",
        confidence=1.0,
    )


@patch("app.retrieval.embeddings.SentenceTransformer")
def test_embedding_model_uses_default_model(mock_sentence_transformer):
    model = MagicMock()
    mock_sentence_transformer.return_value = model

    EmbeddingModel()

    mock_sentence_transformer.assert_called_once_with(MODEL_NAME)


@patch("app.retrieval.embeddings.SentenceTransformer")
def test_dimension_uses_model_embedding_dimension(mock_sentence_transformer):
    model = MagicMock()
    model.get_embedding_dimension.return_value = 384
    mock_sentence_transformer.return_value = model

    embedding_model = EmbeddingModel()

    assert embedding_model.dimension == 384
    model.get_embedding_dimension.assert_called_once()


@patch("app.retrieval.embeddings.SentenceTransformer")
def test_embed_text_rejects_empty_text(mock_sentence_transformer):
    mock_sentence_transformer.return_value = MagicMock()

    embedding_model = EmbeddingModel()

    with pytest.raises(ValueError, match="text must not be empty"):
        embedding_model.embed_text("   ")


@patch("app.retrieval.embeddings.SentenceTransformer")
def test_embed_text_returns_list(mock_sentence_transformer):
    model = MagicMock()
    model.encode.return_value = np.array([0.1, 0.2, 0.3])
    mock_sentence_transformer.return_value = model

    embedding_model = EmbeddingModel()

    result = embedding_model.embed_text("Python")

    assert result == [0.1, 0.2, 0.3]

    model.encode.assert_called_once_with(
        "Python",
        convert_to_numpy=True,
        normalize_embeddings=True,
    )


@patch("app.retrieval.embeddings.SentenceTransformer")
def test_embed_memory_uses_memory_embedding_text(mock_sentence_transformer):
    model = MagicMock()
    mock_sentence_transformer.return_value = model

    embedding_model = EmbeddingModel()

    embedding_model.embed_text = MagicMock(
        return_value=[0.1, 0.2, 0.3]
    )

    memory = make_memory()

    result = embedding_model.embed_memory(memory)

    embedding_model.embed_text.assert_called_once_with(
        memory.to_embedding_text()
    )

    assert result == [0.1, 0.2, 0.3]