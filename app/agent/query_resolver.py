from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class MemoryQuery:
    """
    Structured representation of a user's memory query.
    """

    memory_keys: list[str]
    query_time: datetime | None = None


class QueryResolver:
    """
    Interface for converting a natural-language query
    into a structured memory query.
    """

    def resolve(
        self,
        query: str,
    ) -> MemoryQuery:
        raise NotImplementedError


class StaticQueryResolver(QueryResolver):
    """
    Deterministic resolver used for testing.
    """

    def __init__(
        self,
        memory_query: MemoryQuery,
    ):
        self.memory_query = memory_query

    def resolve(
        self,
        query: str,
    ) -> MemoryQuery:
        if not query.strip():
            raise ValueError(
                "query must not be empty"
            )

        return self.memory_query