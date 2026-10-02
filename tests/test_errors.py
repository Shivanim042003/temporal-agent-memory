import pytest

from app.agent.errors import (
    MemoryAgentError,
    MemoryPersistenceError,
    MemoryQueryError,
    MemoryReasoningError,
    MemoryRetrievalError,
)


def test_memory_agent_error_is_base_exception():
    error = MemoryAgentError(
        "agent failure"
    )

    assert str(error) == "agent failure"


@pytest.mark.parametrize(
    "error_type",
    [
        MemoryQueryError,
        MemoryRetrievalError,
        MemoryReasoningError,
        MemoryPersistenceError,
    ],
)
def test_memory_errors_inherit_from_base(
    error_type,
):
    error = error_type(
        "failure"
    )

    assert isinstance(
        error,
        MemoryAgentError,
    )