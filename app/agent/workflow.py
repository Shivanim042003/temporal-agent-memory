import logging
from datetime import datetime

from langgraph.graph import END, START, StateGraph

from app.agent.query_resolver import QueryResolver
from app.agent.reasoner import AgentReasoner
from app.agent.state import MemoryAgentState
from app.memory.extraction import MemoryExtractionCandidate
from app.memory.extractor import MemoryExtractor
from app.storage.memory_repository import (
    get_at_time,
    insert,
    list_by_key,
)


LOGGER = logging.getLogger(
    "temporal_agent_memory.workflow"
)

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
        LOGGER.info(
            "memory_extraction_started "
            "source_id=%s",
            state.source_id,
        )

        result = extractor.extract(
            state.conversation,
            source_id=state.source_id,
        )

        LOGGER.info(
            "memory_extraction_complete "
            "source_id=%s count=%d",
            state.source_id,
            len(result.memories),
        )

        return {
            "extracted_memories": result.memories,
        }

    return extract_memories


def build_query_resolver_node(
    resolver: QueryResolver,
):
    """
    Resolve a natural-language query into structured
    memory retrieval constraints.
    """

    def resolve_query(
        state: MemoryAgentState,
    ) -> dict:
        result = resolver.resolve(
            state.conversation
        )

        LOGGER.info(
            "memory_query_resolved "
            "keys=%s query_time=%s",
            result.memory_keys,
            result.query_time,
        )

        return {
            "memory_keys": result.memory_keys,
            "query_time": result.query_time,
        }

    return resolve_query


def _is_valid_memory(
    memory: MemoryExtractionCandidate,
) -> bool:
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
    validated_memories = [
        memory
        for memory in state.extracted_memories
        if _is_valid_memory(memory)
    ]

    LOGGER.info(
        "memory_validation_complete "
        "extracted=%d validated=%d rejected=%d",
        len(state.extracted_memories),
        len(validated_memories),
        (
            len(state.extracted_memories)
            - len(validated_memories)
        ),
    )

    return {
        "validated_memories": validated_memories,
    }


def build_write_node(
    db_path,
    id_factory,
):
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

        LOGGER.info(
            "memory_write_complete "
            "count=%d",
            len(written_memories),
        )

        return {
            "written_memories": written_memories,
        }

    return write_memories


def _get_temporal_context(
    memory_key: str,
    *,
    query_time: datetime,
    db_path,
):
    memory = get_at_time(
        memory_key,
        query_time,
        db_path,
    )

    if memory is None:
        return []

    return [memory]


def build_context_node(
    db_path,
):
    def read_memory_context(
        state: MemoryAgentState,
    ) -> dict:
        context_memories = []
        seen_memory_ids = set()

        memory_keys = set(
            state.memory_keys
        )

        for memory_key in memory_keys:
            if state.query_time is not None:
                memories = _get_temporal_context(
                    memory_key,
                    query_time=state.query_time,
                    db_path=db_path,
                )
            else:
                memories = list_by_key(
                    memory_key,
                    db_path,
                )

            for memory in memories:
                if memory.memory_id in seen_memory_ids:
                    continue

                seen_memory_ids.add(
                    memory.memory_id
                )

                context_memories.append(
                    memory
                )

        LOGGER.info(
            "memory_context_retrieved "
            "keys=%s count=%d",
            list(memory_keys),
            len(context_memories),
        )

        return {
            "context_memories": context_memories,
        }

    return read_memory_context


def build_reason_node(
    reasoner: AgentReasoner,
):
    def reason(
        state: MemoryAgentState,
    ) -> dict:
        answer = reasoner.answer(
            state.conversation,
            memories=state.context_memories,
        )

        LOGGER.info(
            "memory_reasoning_complete "
            "memory_count=%d",
            len(state.context_memories),
        )

        return {
            "answer": answer,
        }

    return reason


def build_memory_workflow(
    extractor: MemoryExtractor,
    *,
    reasoner: AgentReasoner | None = None,
    query_resolver: QueryResolver | None = None,
    db_path=None,
    id_factory=None,
):
    """
    Build the main memory ingestion and reasoning workflow.

    With database configuration:

        extract
          ->
        validate
          ->
        write
          ->
        context
          ->
        reason

    If query_resolver is supplied:

        resolve_query
          ->
        extract
          ->
        validate
          ->
        write
          ->
        context
          ->
        reason
    """

    graph = StateGraph(
        MemoryAgentState
    )

    if query_resolver is not None:
        graph.add_node(
            "resolve_query",
            build_query_resolver_node(
                query_resolver
            ),
        )

    graph.add_node(
        "extract",
        build_extract_node(
            extractor
        ),
    )

    graph.add_node(
        "validate",
        validate_memories,
    )

    if query_resolver is not None:
        graph.add_edge(
            START,
            "resolve_query",
        )

        graph.add_edge(
            "resolve_query",
            "extract",
        )
    else:
        graph.add_edge(
            START,
            "extract",
        )

    graph.add_edge(
        "extract",
        "validate",
    )

    if (
        db_path is not None
        and id_factory is not None
    ):
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
                db_path
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

        if reasoner is not None:
            graph.add_node(
                "reason",
                build_reason_node(
                    reasoner
                ),
            )

            graph.add_edge(
                "context",
                "reason",
            )

            graph.add_edge(
                "reason",
                END,
            )
        else:
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


def build_memory_query_workflow(
    resolver: QueryResolver,
    *,
    reasoner: AgentReasoner,
    db_path,
):
    """
    Build a read-only memory reasoning workflow.

    resolve_query
        ->
    context
        ->
    reason
        ->
    END
    """

    graph = StateGraph(
        MemoryAgentState
    )

    graph.add_node(
        "resolve_query",
        build_query_resolver_node(
            resolver
        ),
    )

    graph.add_node(
        "context",
        build_context_node(
            db_path
        ),
    )

    graph.add_node(
        "reason",
        build_reason_node(
            reasoner
        ),
    )

    graph.add_edge(
        START,
        "resolve_query",
    )

    graph.add_edge(
        "resolve_query",
        "context",
    )

    graph.add_edge(
        "context",
        "reason",
    )

    graph.add_edge(
        "reason",
        END,
    )

    return graph.compile()