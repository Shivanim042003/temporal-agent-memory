from datetime import datetime, timezone

from app.memory.extraction import (
    ExtractionEvidence,
    MemoryExtractionCandidate,
    MemoryExtractionResult,
)
from app.memory.extractor import StaticMemoryExtractor
from app.memory.ingestion import MemoryIngestionService
from app.memory.models import MemoryType
from app.storage.database import initialize_database
from app.storage.memory_repository import get


def test_end_to_end_memory_extraction_pipeline(tmp_path):
    database_path = tmp_path / "memory.db"
    initialize_database(database_path)

    extraction_result = MemoryExtractionResult(
        memories=[
            MemoryExtractionCandidate(
                subject="user",
                attribute="programming_language",
                value="Node.js",
                memory_type=MemoryType.SKILL,
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
                confidence=0.96,
                evidence=ExtractionEvidence.EXPLICIT,
                source_id="conversation-100",
            ),
            MemoryExtractionCandidate(
                subject="user",
                attribute="programming_language",
                value="Python",
                memory_type=MemoryType.SKILL,
                valid_from=datetime(
                    2026,
                    6,
                    1,
                    tzinfo=timezone.utc,
                ),
                confidence=0.94,
                evidence=ExtractionEvidence.EXPLICIT,
                source_id="conversation-100",
            ),
        ]
    )

    extractor = StaticMemoryExtractor(
        result=extraction_result
    )

    ids = iter(
        [
            "memory-node",
            "memory-python",
        ]
    )

    ingestion = MemoryIngestionService(
        extractor=extractor,
        id_factory=lambda: next(ids),
    )

    memories = ingestion.ingest(
        """
        I used Node.js earlier this year.
        Since June I have been using Python.
        """,
        source_id="conversation-100",
        db_path=database_path,
    )

    assert len(memories) == 2

    assert memories[0].memory_id == "memory-node"
    assert memories[0].value == "Node.js"

    assert memories[1].memory_id == "memory-python"
    assert memories[1].value == "Python"

    stored_node = get(
        "memory-node",
        database_path,
    )

    stored_python = get(
        "memory-python",
        database_path,
    )

    assert stored_node is not None
    assert stored_python is not None

    assert stored_node.value == "Node.js"
    assert stored_python.value == "Python"

    assert stored_node.valid_to == datetime(
        2026,
        6,
        1,
        tzinfo=timezone.utc,
    )

    assert stored_python.valid_to is None


def test_end_to_end_pipeline_preserves_provenance(
    tmp_path,
):
    database_path = tmp_path / "memory.db"
    initialize_database(database_path)

    extraction_result = MemoryExtractionResult(
        memories=[
            MemoryExtractionCandidate(
                subject="user",
                attribute="goal",
                value="Build an LLM system",
                memory_type=MemoryType.GOAL,
                valid_from=datetime(
                    2026,
                    9,
                    1,
                    tzinfo=timezone.utc,
                ),
                confidence=0.91,
                evidence=ExtractionEvidence.INFERRED,
                source_id="conversation-200",
            )
        ]
    )

    extractor = StaticMemoryExtractor(
        result=extraction_result
    )

    ingestion = MemoryIngestionService(
        extractor=extractor,
        id_factory=lambda: "memory-goal",
    )

    memories = ingestion.ingest(
        "I want to build an LLM system.",
        source_id="conversation-200",
        db_path=database_path,
    )

    assert len(memories) == 1

    memory = memories[0]

    assert memory.source_id == "conversation-200"
    assert memory.evidence_type.value == "inferred"
    assert memory.confidence == 0.91
    assert memory.memory_type == MemoryType.GOAL

    stored = get(
        "memory-goal",
        database_path,
    )

    assert stored is not None
    assert stored.source_id == "conversation-200"
    assert stored.evidence_type.value == "inferred"