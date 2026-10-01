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


def make_candidate(
    *,
    value: str,
    source_id: str = "conversation-1",
):
    return MemoryExtractionCandidate(
        subject="user",
        attribute="programming_language",
        value=value,
        memory_type=MemoryType.SKILL,
        valid_from=datetime(
            2026,
            9,
            1,
            tzinfo=timezone.utc,
        ),
        confidence=0.95,
        evidence=ExtractionEvidence.EXPLICIT,
        source_id=source_id,
    )


def test_ingestion_stores_extracted_memory(tmp_path):
    database_path = tmp_path / "memory.db"
    initialize_database(database_path)

    extractor = StaticMemoryExtractor(
        MemoryExtractionResult(
            memories=[
                make_candidate(
                    value="Python"
                )
            ]
        )
    )

    ids = iter(["memory-1"])

    service = MemoryIngestionService(
        extractor=extractor,
        id_factory=lambda: next(ids),
    )

    memories = service.ingest(
        "I am learning Python.",
        source_id="conversation-1",
        db_path=database_path,
    )

    assert len(memories) == 1
    assert memories[0].memory_id == "memory-1"
    assert memories[0].value == "Python"

    stored = get(
        "memory-1",
        database_path,
    )

    assert stored is not None
    assert stored.value == "Python"


def test_ingestion_stores_multiple_memories(tmp_path):
    database_path = tmp_path / "memory.db"
    initialize_database(database_path)

    extractor = StaticMemoryExtractor(
        MemoryExtractionResult(
            memories=[
                make_candidate(value="Python"),
                make_candidate(value="Java"),
            ]
        )
    )

    ids = iter(
        [
            "memory-1",
            "memory-2",
        ]
    )

    service = MemoryIngestionService(
        extractor=extractor,
        id_factory=lambda: next(ids),
    )

    memories = service.ingest(
        "I use Python and Java.",
        source_id="conversation-1",
        db_path=database_path,
    )

    assert [
        memory.memory_id
        for memory in memories
    ] == [
        "memory-1",
        "memory-2",
    ]

    assert [
        memory.value
        for memory in memories
    ] == [
        "Python",
        "Java",
    ]


def test_ingestion_returns_empty_list_when_no_memories(
    tmp_path,
):
    database_path = tmp_path / "memory.db"
    initialize_database(database_path)

    extractor = StaticMemoryExtractor(
        MemoryExtractionResult()
    )

    service = MemoryIngestionService(
        extractor=extractor,
        id_factory=lambda: "unused",
    )

    memories = service.ingest(
        "Hello there.",
        source_id="conversation-1",
        db_path=database_path,
    )

    assert memories == []


def test_ingestion_preserves_source_id(
    tmp_path,
):
    database_path = tmp_path / "memory.db"
    initialize_database(database_path)

    extractor = StaticMemoryExtractor(
        MemoryExtractionResult(
            memories=[
                make_candidate(
                    value="Python",
                    source_id="conversation-original",
                )
            ]
        )
    )

    service = MemoryIngestionService(
        extractor=extractor,
        id_factory=lambda: "memory-1",
    )

    memories = service.ingest(
        "I use Python.",
        source_id="conversation-original",
        db_path=database_path,
    )

    assert memories[0].source_id == (
        "conversation-original"
    )

    stored = get(
        "memory-1",
        database_path,
    )

    assert stored is not None
    assert stored.source_id == (
        "conversation-original"
    )