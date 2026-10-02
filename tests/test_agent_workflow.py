from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from app.agent.query_resolver import (
    MemoryQuery,
    StaticQueryResolver,
)
from app.agent.reasoner import StaticAgentReasoner
from app.agent.state import MemoryAgentState
from app.agent.workflow import (
    build_memory_query_workflow,
    build_memory_workflow,
)
from app.memory.extraction import (
    ExtractionEvidence,
    MemoryExtractionCandidate,
    MemoryExtractionResult,
)
from app.memory.extractor import (
    StaticMemoryExtractor,
)
from app.memory.models import (
    MemoryType,
)
from app.storage.database import (
    initialize_database,
)
from app.storage.memory_repository import (
    list_by_key,
)


def _candidate(
    *,
    subject: str,
    attribute: str,
    value: str,
    valid_from: datetime,
    valid_to: datetime | None = None,
    confidence: float = 0.95,
    source_id: str = "conversation-1",
):
    return MemoryExtractionCandidate(
        subject=subject,
        attribute=attribute,
        value=value,
        memory_type=MemoryType.SKILL,
        valid_from=valid_from,
        valid_to=valid_to,
        confidence=confidence,
        evidence=ExtractionEvidence.EXPLICIT,
        source_id=source_id,
    )


def _extractor(
    candidates,
):
    return StaticMemoryExtractor(
        MemoryExtractionResult(
            memories=candidates
        )
    )


def _db_path(
    tmp_path: Path,
):
    db_path = tmp_path / "memory.db"

    initialize_database(
        db_path
    )

    return db_path


def test_workflow_extracts_writes_reads_and_reasons(
    tmp_path,
):
    db_path = _db_path(
        tmp_path
    )

    candidate = _candidate(
        subject="user",
        attribute="language",
        value="Python",
        valid_from=datetime(
            2026,
            9,
            1,
            tzinfo=timezone.utc,
        ),
    )

    workflow = build_memory_workflow(
        _extractor([candidate]),
        reasoner=StaticAgentReasoner(),
        db_path=db_path,
        id_factory=lambda: str(uuid4()),
    )

    result = workflow.invoke(
        MemoryAgentState(
            conversation="I use Python.",
            source_id="conversation-1",
            memory_keys=[
                "user:language"
            ],
        )
    )

    assert len(
        result["extracted_memories"]
    ) == 1

    assert len(
        result["validated_memories"]
    ) == 1

    assert len(
        result["written_memories"]
    ) == 1

    assert len(
        result["context_memories"]
    ) == 1

    assert (
        "Python"
        in result["answer"]
    )


def test_workflow_returns_no_memory_when_context_is_empty(
    tmp_path,
):
    db_path = _db_path(
        tmp_path
    )

    workflow = build_memory_workflow(
        _extractor([]),
        reasoner=StaticAgentReasoner(),
        db_path=db_path,
        id_factory=lambda: str(uuid4()),
    )

    result = workflow.invoke(
        MemoryAgentState(
            conversation="What do I use?",
            source_id="conversation-1",
            memory_keys=[
                "user:language"
            ],
        )
    )

    assert (
        result["context_memories"]
        == []
    )

    assert (
        result["answer"]
        == "No relevant memories found."
    )


def test_workflow_retrieves_historical_nodejs(
    tmp_path,
):
    db_path = _db_path(
        tmp_path
    )

    node_candidate = _candidate(
        subject="user",
        attribute="language",
        value="Node.js",
        valid_from=datetime(
            2026,
            1,
            1,
            tzinfo=timezone.utc,
        ),
        valid_to=datetime(
            2026,
            6,
            1,
            tzinfo=timezone.utc,
        ),
    )

    workflow = build_memory_workflow(
        _extractor(
            [node_candidate]
        ),
        reasoner=StaticAgentReasoner(),
        db_path=db_path,
        id_factory=lambda: str(uuid4()),
    )

    workflow.invoke(
        MemoryAgentState(
            conversation="I use Node.js.",
            source_id="conversation-1",
            memory_keys=[
                "user:language"
            ],
        )
    )

    resolver = StaticQueryResolver(
        MemoryQuery(
            memory_keys=[
                "user:language"
            ],
            query_time=datetime(
                2026,
                3,
                1,
                tzinfo=timezone.utc,
            ),
        )
    )

    query_workflow = build_memory_query_workflow(
        resolver,
        reasoner=StaticAgentReasoner(),
        db_path=db_path,
    )

    result = query_workflow.invoke(
        MemoryAgentState(
            conversation="What language did I use?",
            source_id="query-1",
        )
    )

    assert len(
        result["context_memories"]
    ) == 1

    assert (
        result["context_memories"][0].value
        == "Node.js"
    )


def test_workflow_retrieves_historical_go(
    tmp_path,
):
    db_path = _db_path(
        tmp_path
    )

    candidates = [
        _candidate(
            subject="user",
            attribute="language",
            value="Node.js",
            valid_from=datetime(
                2026,
                1,
                1,
                tzinfo=timezone.utc,
            ),
            valid_to=datetime(
                2026,
                6,
                1,
                tzinfo=timezone.utc,
            ),
        ),
        _candidate(
            subject="user",
            attribute="language",
            value="Go",
            valid_from=datetime(
                2026,
                6,
                1,
                tzinfo=timezone.utc,
            ),
            valid_to=datetime(
                2026,
                9,
                1,
                tzinfo=timezone.utc,
            ),
        ),
    ]

    workflow = build_memory_workflow(
        _extractor(candidates),
        reasoner=StaticAgentReasoner(),
        db_path=db_path,
        id_factory=lambda: str(uuid4()),
    )

    workflow.invoke(
        MemoryAgentState(
            conversation="My languages changed.",
            source_id="conversation-1",
            memory_keys=[
                "user:language"
            ],
        )
    )

    resolver = StaticQueryResolver(
        MemoryQuery(
            memory_keys=[
                "user:language"
            ],
            query_time=datetime(
                2026,
                7,
                1,
                tzinfo=timezone.utc,
            ),
        )
    )

    query_workflow = build_memory_query_workflow(
        resolver,
        reasoner=StaticAgentReasoner(),
        db_path=db_path,
    )

    result = query_workflow.invoke(
        MemoryAgentState(
            conversation="What language was I using?",
            source_id="query-1",
        )
    )

    assert (
        result["context_memories"][0].value
        == "Go"
    )


def test_workflow_retrieves_current_python(
    tmp_path,
):
    db_path = _db_path(
        tmp_path
    )

    candidates = [
        _candidate(
            subject="user",
            attribute="language",
            value="Python",
            valid_from=datetime(
                2026,
                9,
                1,
                tzinfo=timezone.utc,
            ),
        ),
    ]

    workflow = build_memory_workflow(
        _extractor(candidates),
        reasoner=StaticAgentReasoner(),
        db_path=db_path,
        id_factory=lambda: str(uuid4()),
    )

    workflow.invoke(
        MemoryAgentState(
            conversation="I now use Python.",
            source_id="conversation-1",
            memory_keys=[
                "user:language"
            ],
        )
    )

    resolver = StaticQueryResolver(
        MemoryQuery(
            memory_keys=[
                "user:language"
            ],
            query_time=datetime(
                2026,
                10,
                1,
                tzinfo=timezone.utc,
            ),
        )
    )

    query_workflow = build_memory_query_workflow(
        resolver,
        reasoner=StaticAgentReasoner(),
        db_path=db_path,
    )

    result = query_workflow.invoke(
        MemoryAgentState(
            conversation="What do I use now?",
            source_id="query-1",
        )
    )

    assert (
        result["context_memories"][0].value
        == "Python"
    )


def test_workflow_respects_half_open_interval(
    tmp_path,
):
    db_path = _db_path(
        tmp_path
    )

    candidates = [
        _candidate(
            subject="user",
            attribute="language",
            value="Node.js",
            valid_from=datetime(
                2026,
                1,
                1,
                tzinfo=timezone.utc,
            ),
            valid_to=datetime(
                2026,
                6,
                1,
                tzinfo=timezone.utc,
            ),
        ),
        _candidate(
            subject="user",
            attribute="language",
            value="Go",
            valid_from=datetime(
                2026,
                6,
                1,
                tzinfo=timezone.utc,
            ),
        ),
    ]

    workflow = build_memory_workflow(
        _extractor(candidates),
        reasoner=StaticAgentReasoner(),
        db_path=db_path,
        id_factory=lambda: str(uuid4()),
    )

    workflow.invoke(
        MemoryAgentState(
            conversation="Language history.",
            source_id="conversation-1",
            memory_keys=[
                "user:language"
            ],
        )
    )

    resolver = StaticQueryResolver(
        MemoryQuery(
            memory_keys=[
                "user:language"
            ],
            query_time=datetime(
                2026,
                6,
                1,
                tzinfo=timezone.utc,
            ),
        )
    )

    query_workflow = build_memory_query_workflow(
        resolver,
        reasoner=StaticAgentReasoner(),
        db_path=db_path,
    )

    result = query_workflow.invoke(
        MemoryAgentState(
            conversation="What language?",
            source_id="query-1",
        )
    )

    assert len(
        result["context_memories"]
    ) == 1

    assert (
        result["context_memories"][0].value
        == "Go"
    )


def test_workflow_supports_cross_session_memory(
    tmp_path,
):
    db_path = _db_path(
        tmp_path
    )

    candidate = _candidate(
        subject="user",
        attribute="language",
        value="Python",
        valid_from=datetime(
            2026,
            9,
            1,
            tzinfo=timezone.utc,
        ),
        source_id="session-1",
    )

    write_workflow = build_memory_workflow(
        _extractor([candidate]),
        db_path=db_path,
        id_factory=lambda: str(uuid4()),
    )

    write_workflow.invoke(
        MemoryAgentState(
            conversation="I use Python.",
            source_id="session-1",
            memory_keys=[
                "user:language"
            ],
        )
    )

    resolver = StaticQueryResolver(
        MemoryQuery(
            memory_keys=[
                "user:language"
            ],
            query_time=datetime(
                2026,
                10,
                1,
                tzinfo=timezone.utc,
            ),
        )
    )

    query_workflow = build_memory_query_workflow(
        resolver,
        reasoner=StaticAgentReasoner(),
        db_path=db_path,
    )

    result = query_workflow.invoke(
        MemoryAgentState(
            conversation="What do I use?",
            source_id="session-2",
        )
    )

    assert (
        result["context_memories"][0].value
        == "Python"
    )


def test_query_resolver_populates_state(
    tmp_path,
):
    db_path = _db_path(
        tmp_path
    )

    query_time = datetime(
        2026,
        10,
        1,
        tzinfo=timezone.utc,
    )

    resolver = StaticQueryResolver(
        MemoryQuery(
            memory_keys=[
                "user:language"
            ],
            query_time=query_time,
        )
    )

    workflow = build_memory_query_workflow(
        resolver,
        reasoner=StaticAgentReasoner(),
        db_path=db_path,
    )

    result = workflow.invoke(
        MemoryAgentState(
            conversation="What language?",
            source_id="query-1",
        )
    )

    assert result["memory_keys"] == [
        "user:language"
    ]

    assert (
        result["query_time"]
        == query_time
    )


def test_read_only_query_does_not_write(
    tmp_path,
):
    db_path = _db_path(
        tmp_path
    )

    candidate = _candidate(
        subject="user",
        attribute="language",
        value="Python",
        valid_from=datetime(
            2026,
            9,
            1,
            tzinfo=timezone.utc,
        ),
    )

    write_workflow = build_memory_workflow(
        _extractor([candidate]),
        db_path=db_path,
        id_factory=lambda: str(uuid4()),
    )

    write_workflow.invoke(
        MemoryAgentState(
            conversation="I use Python.",
            source_id="session-1",
            memory_keys=[
                "user:language"
            ],
        )
    )

    before = list_by_key(
        "user:language",
        db_path,
    )

    resolver = StaticQueryResolver(
        MemoryQuery(
            memory_keys=[
                "user:language"
            ],
            query_time=datetime(
                2026,
                10,
                1,
                tzinfo=timezone.utc,
            ),
        )
    )

    query_workflow = build_memory_query_workflow(
        resolver,
        reasoner=StaticAgentReasoner(),
        db_path=db_path,
    )

    query_workflow.invoke(
        MemoryAgentState(
            conversation="What do I use?",
            source_id="session-2",
        )
    )

    after = list_by_key(
        "user:language",
        db_path,
    )

    assert len(before) == len(after)


def test_read_only_historical_query(
    tmp_path,
):
    db_path = _db_path(
        tmp_path
    )

    candidates = [
        _candidate(
            subject="user",
            attribute="language",
            value="Node.js",
            valid_from=datetime(
                2026,
                1,
                1,
                tzinfo=timezone.utc,
            ),
            valid_to=datetime(
                2026,
                6,
                1,
                tzinfo=timezone.utc,
            ),
        ),
        _candidate(
            subject="user",
            attribute="language",
            value="Go",
            valid_from=datetime(
                2026,
                6,
                1,
                tzinfo=timezone.utc,
            ),
            valid_to=datetime(
                2026,
                9,
                1,
                tzinfo=timezone.utc,
            ),
        ),
    ]

    write_workflow = build_memory_workflow(
        _extractor(candidates),
        db_path=db_path,
        id_factory=lambda: str(uuid4()),
    )

    write_workflow.invoke(
        MemoryAgentState(
            conversation="Language history.",
            source_id="session-1",
            memory_keys=[
                "user:language"
            ],
        )
    )

    resolver = StaticQueryResolver(
        MemoryQuery(
            memory_keys=[
                "user:language"
            ],
            query_time=datetime(
                2026,
                3,
                1,
                tzinfo=timezone.utc,
            ),
        )
    )

    query_workflow = build_memory_query_workflow(
        resolver,
        reasoner=StaticAgentReasoner(),
        db_path=db_path,
    )

    result = query_workflow.invoke(
        MemoryAgentState(
            conversation="What did I use?",
            source_id="session-2",
        )
    )

    assert (
        result["context_memories"][0].value
        == "Node.js"
    )


def test_read_only_current_query(
    tmp_path,
):
    db_path = _db_path(
        tmp_path
    )

    candidate = _candidate(
        subject="user",
        attribute="language",
        value="Python",
        valid_from=datetime(
            2026,
            9,
            1,
            tzinfo=timezone.utc,
        ),
    )

    write_workflow = build_memory_workflow(
        _extractor([candidate]),
        db_path=db_path,
        id_factory=lambda: str(uuid4()),
    )

    write_workflow.invoke(
        MemoryAgentState(
            conversation="I use Python.",
            source_id="session-1",
            memory_keys=[
                "user:language"
            ],
        )
    )

    resolver = StaticQueryResolver(
        MemoryQuery(
            memory_keys=[
                "user:language"
            ],
            query_time=datetime(
                2026,
                10,
                1,
                tzinfo=timezone.utc,
            ),
        )
    )

    query_workflow = build_memory_query_workflow(
        resolver,
        reasoner=StaticAgentReasoner(),
        db_path=db_path,
    )

    result = query_workflow.invoke(
        MemoryAgentState(
            conversation="What do I use now?",
            source_id="session-2",
        )
    )

    assert (
        result["context_memories"][0].value
        == "Python"
    )