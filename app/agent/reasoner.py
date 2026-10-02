from abc import ABC, abstractmethod

from app.memory.models import Memory


class AgentReasoner(ABC):
    @abstractmethod
    def answer(
        self,
        query: str,
        *,
        memories: list[Memory],
    ) -> str:
        raise NotImplementedError


class StaticAgentReasoner(AgentReasoner):
    def answer(
        self,
        query: str,
        *,
        memories: list[Memory],
    ) -> str:
        if not query.strip():
            raise ValueError(
                "query must not be empty"
            )

        if not memories:
            return "No relevant memories found."

        memory_lines = [
            (
                f"{memory.subject} "
                f"{memory.attribute}: "
                f"{memory.value}"
            )
            for memory in memories
        ]

        return (
            f"Query: {query}\n"
            f"Memory context:\n"
            + "\n".join(memory_lines)
        )