import logging

from app.agent.query_resolver import (
    MemoryQuery,
    StaticQueryResolver,
)
from app.agent.reasoner import StaticAgentReasoner
from app.agent.state import MemoryAgentState
from app.agent.workflow import (
    build_memory_query_workflow,
)
from app.storage.database import (
    initialize_database,
)


def test_query_resolution_is_logged(
    tmp_path,
    caplog,
):
    db_path = tmp_path / "memory.db"

    initialize_database(
        db_path
    )

    resolver = StaticQueryResolver(
        MemoryQuery(
            memory_keys=[
                "user:language"
            ],
        )
    )

    workflow = build_memory_query_workflow(
        resolver,
        reasoner=StaticAgentReasoner(),
        db_path=db_path,
    )

    with caplog.at_level(
        logging.INFO,
        logger=(
            "temporal_agent_memory.workflow"
        ),
    ):
        workflow.invoke(
            MemoryAgentState(
                conversation="What language?",
                source_id="query-1",
            )
        )

    assert (
        "memory_query_resolved"
        in caplog.text
    )


def test_context_retrieval_is_logged(
    tmp_path,
    caplog,
):
    db_path = tmp_path / "memory.db"

    initialize_database(
        db_path
    )

    resolver = StaticQueryResolver(
        MemoryQuery(
            memory_keys=[
                "user:language"
            ],
        )
    )

    workflow = build_memory_query_workflow(
        resolver,
        reasoner=StaticAgentReasoner(),
        db_path=db_path,
    )

    with caplog.at_level(
        logging.INFO,
        logger=(
            "temporal_agent_memory.workflow"
        ),
    ):
        workflow.invoke(
            MemoryAgentState(
                conversation="What language?",
                source_id="query-1",
            )
        )

    assert (
        "memory_context_retrieved"
        in caplog.text
    )


def test_reasoning_completion_is_logged(
    tmp_path,
    caplog,
):
    db_path = tmp_path / "memory.db"

    initialize_database(
        db_path
    )

    resolver = StaticQueryResolver(
        MemoryQuery(
            memory_keys=[]
        )
    )

    workflow = build_memory_query_workflow(
        resolver,
        reasoner=StaticAgentReasoner(),
        db_path=db_path,
    )

    with caplog.at_level(
        logging.INFO,
        logger=(
            "temporal_agent_memory.workflow"
        ),
    ):
        workflow.invoke(
            MemoryAgentState(
                conversation="What do you remember?",
                source_id="query-1",
            )
        )

    assert (
        "memory_reasoning_complete"
        in caplog.text
    )