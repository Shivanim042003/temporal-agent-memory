class MemoryAgentError(Exception):
    """Base exception for the temporal memory agent."""

    pass


class MemoryQueryError(MemoryAgentError):
    """Raised when a memory query cannot be resolved."""

    pass


class MemoryRetrievalError(MemoryAgentError):
    """Raised when memory retrieval fails."""

    pass


class MemoryReasoningError(MemoryAgentError):
    """Raised when agent reasoning fails."""

    pass


class MemoryPersistenceError(MemoryAgentError):
    """Raised when memory persistence fails."""

    pass