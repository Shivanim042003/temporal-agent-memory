from langgraph.graph import END, START, StateGraph

from app.agent.state import MemoryAgentState
from app.memory.extraction import MemoryExtractionCandidate
from app.memory.extractor import MemoryExtractor
from app.storage.memory_repository import get, insert, list_by_key


MIN_MEMORY_CONFIDENCE = 0.5


def build_extract_node(
    extractor: MemoryExtractor,
):
    """
    Create a LangGraph node that extracts memories
    using the supplied MemoryExtractor.
    """

    def extract_memories(
        state: MemoryAgentState,
    ) -> dict:
        result = extractor.extract(
            state.conversation,
            source_id=state.source_id,
        )

        return {
            "extracted_memories": result.memories,
        }

    return extract_memories


def _is_valid_memory(
    memory: MemoryExtractionCandidate,
) -> bool:
    """
    Check whether an extracted memory is safe to pass
    to the memory-writing stage.
    """

    if memory.confidence < MIN_MEMORY_CONFIDENCE:
        return False

    if not memory.subject.strip():
        return False

    if not memory.attribute.strip():
        return False

    if not memory.value.strip():
        return False

    if (
        memory.valid_to is not None
        and memory.valid_to <= memory.valid_from
    ):
        return False

    return True


def validate_memories(
    state: MemoryAgentState,
) -> dict:
    """
    Validate extracted memory candidates.

    Invalid candidates are rejected and never reach
    the memory-writing stage.
    """

    validated_memories = [
        memory
        for memory in state.extracted_memories
        if _is_valid_memory(memory)
    ]

    return {
        "validated_memories": validated_memories,
    }


def build_write_node(
    db_path,
    id_factory,
):
    """
    Create a LangGraph node that converts validated
    candidates into Memory objects and persists them.
    """

    def write_memories(
        state: MemoryAgentState,
    ) -> dict:
        written_memories = []

        for candidate in state.validated_memories:
            memory = candidate.to_memory(
                memory_id=id_factory(),
            )

            insert(
                memory,
                db_path,
            )

            written_memories.append(memory)

        return {
            "written_memories": written_memories,
        }

    return write_memories


def build_context_node(
    db_path,
):
    """
    Create a LangGraph node that reads memory context
    from SQLite using the keys of memories written
    during the current workflow execution.
    """

    def read_memory_context(
        state: MemoryAgentState,
    ) -> dict:
        context_memories = []
        seen_memory_ids = set()

        memory_keys = {
            memory.memory_key
            for memory in state.written_memories
        }

        for memory_key in memory_keys:
            memories = list_by_key(
                memory_key,
                db_path,
            )

            for memory in memories:
                if memory.memory_id in seen_memory_ids:
                    continue

                seen_memory_ids.add(memory.memory_id)
                context_memories.append(memory)

        return {
            "context_memories": context_memories,
        }

    return read_memory_context


def build_memory_workflow(
    extractor: MemoryExtractor,
    *,
    db_path=None,
    id_factory=None,
):
    """
    Build the LangGraph memory workflow.

    When db_path and id_factory are supplied:

        extract -> validate -> write -> context

    Otherwise:

        extract -> validate
    """

    graph = StateGraph(MemoryAgentState)

    graph.add_node(
        "extract",
        build_extract_node(extractor),
    )

    graph.add_node(
        "validate",
        validate_memories,
    )

    graph.add_edge(
        START,
        "extract",
    )

    graph.add_edge(
        "extract",
        "validate",
    )

    if db_path is not None and id_factory is not None:
        graph.add_node(
            "write",
            build_write_node(
                db_path,
                id_factory,
            ),
        )

        graph.add_node(
            "context",
            build_context_node(
                db_path,
            ),
        )

        graph.add_edge(
            "validate",
            "write",
        )

        graph.add_edge(
            "write",
            "context",
        )

        graph.add_edge(
            "context",
            END,
        )
    else:
        graph.add_edge(
            "validate",
            END,
        )

    return graph.compile()