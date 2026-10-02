from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException

from app.agent.query_resolver import (
    MemoryQuery,
    StaticQueryResolver,
)
from app.agent.reasoner import StaticAgentReasoner
from app.agent.state import MemoryAgentState
from app.agent.workflow import (
    build_memory_query_workflow,
)
from app.api.schemas import (
    MemoryQueryRequest,
    MemoryQueryResponse,
)
from app.storage.database import (
    DEFAULT_DB_PATH,
    get_connection,
    initialize_database,
)


DB_PATH = Path(
    DEFAULT_DB_PATH
)

initialize_database(
    DB_PATH
)

app = FastAPI(
    title="Temporal Agent Memory",
    version="0.1.0",
)


@app.get("/health")
def health():
    """
    Verify application and SQLite connectivity.
    """

    try:
        with get_connection(
            DB_PATH
        ) as connection:
            connection.execute(
                "SELECT 1"
            )

        return {
            "status": "ok",
            "database": "ok",
        }

    except Exception:
        return {
            "status": "degraded",
            "database": "unavailable",
        }


@app.post(
    "/memory/query",
    response_model=MemoryQueryResponse,
)
def query_memory(
    request: MemoryQueryRequest,
):
    """
    Query existing temporal memories.
    """

    resolver = StaticQueryResolver(
        MemoryQuery(
            memory_keys=[],
            query_time=datetime.now(
                timezone.utc
            ),
        )
    )

    try:
        workflow = build_memory_query_workflow(
            resolver,
            reasoner=StaticAgentReasoner(),
            db_path=DB_PATH,
        )

        result = workflow.invoke(
            MemoryAgentState(
                conversation=request.query,
                source_id=request.source_id,
            )
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Memory query failed.",
        ) from exc

    return MemoryQueryResponse(
        answer=result["answer"] or "",
        memories=[
            memory.model_dump(
                mode="json"
            )
            for memory in result[
                "context_memories"
            ]
        ],
        query_time=result[
            "query_time"
        ],
    )