from datetime import datetime, timezone

import pytest

from app.agent.state import MemoryAgentState
from app.agent.workflow import build_memory_workflow
from app.memory.extraction import (
    ExtractionEvidence,
    MemoryExtractionCandidate,
    MemoryExtractionResult,
)
from app.memory.extractor import StaticMemoryExtractor
from app.memory.models import MemoryType
from app.storage.database import initialize_database
from app.storage.memory_repository import get


def make_candidate(
    *,
    value: str = "Python",
    confidence: float = 0.95,
    valid_to: datetime | None = None,
) -> MemoryExtractionCandidate:
    return MemoryExtractionCandidate(
        subject="user",
        attribute="language",
        value=value,
        memory_type=MemoryType.SKILL,
        valid_from=datetime(
            2026,
            1,
            1,
            tzinfo=timezone.utc,
        ),
        valid_to=valid_to,
        confidence=confidence,
        evidence=ExtractionEvidence.EXPLICIT,
        source_id="conversation-1",
    )


def run_workflow(
    candidates: list[MemoryExtractionCandidate],
):
    extractor = StaticMemoryExtractor(
        MemoryExtractionResult(
            memories=candidates,
        )
    )

    workflow = build_memory_workflow(extractor)

    return workflow.invoke(
        MemoryAgentState(
            conversation="I use Python",
            source_id="conversation-1",
        )
    )


def test_valid_memory_is_accepted():
    result = run_workflow(
        [
            make_candidate(),
        ]
    )

    assert len(result["extracted_memories"]) == 1
    assert len(result["validated_memories"]) == 1


def test_low_confidence_memory_is_rejected():
    result = run_workflow(
        [
            make_candidate(confidence=0.49),
        ]
    )

    assert len(result["extracted_memories"]) == 1
    assert result["validated_memories"] == []


def test_boundary_confidence_is_accepted():
    result = run_workflow(
        [
            make_candidate(confidence=0.5),
        ]
    )

    assert len(result["validated_memories"]) == 1


def test_multiple_candidates_are_validated_independently():
    result = run_workflow(
        [
            make_candidate(
                value="Python",
                confidence=0.95,
            ),
            make_candidate(
                value="Java",
                confidence=0.49,
            ),
        ]
    )

    assert len(result["extracted_memories"]) == 2
    assert len(result["validated_memories"]) == 1
    assert result["validated_memories"][0].value == "Python"


def test_valid_temporal_interval_is_accepted():
    result = run_workflow(
        [
            make_candidate(
                valid_to=datetime(
                    2026,
                    6,
                    1,
                    tzinfo=timezone.utc,
                ),
            ),
        ]
    )

    assert len(result["validated_memories"]) == 1


def test_empty_extraction_produces_empty_validation():
    result = run_workflow([])

    assert result["extracted_memories"] == []
    assert result["validated_memories"] == []


def test_extractor_failure_propagates():
    class FailingExtractor(StaticMemoryExtractor):
        def extract(
            self,
            conversation: str,
            *,
            source_id: str,
        ):
            raise RuntimeError("extraction failed")

    workflow = build_memory_workflow(
        FailingExtractor()
    )

    with pytest.raises(RuntimeError, match="extraction failed"):
        workflow.invoke(
            MemoryAgentState(
                conversation="I use Python",
                source_id="conversation-1",
            )
        )


def test_validated_memory_is_written_to_sqlite(tmp_path):
    db_path = tmp_path / "workflow.db"

    initialize_database(db_path)

    extractor = StaticMemoryExtractor(
        MemoryExtractionResult(
            memories=[
                make_candidate(),
            ]
        )
    )

    workflow = build_memory_workflow(
        extractor,
        db_path=db_path,
        id_factory=lambda: "memory-1",
    )

    result = workflow.invoke(
        MemoryAgentState(
            conversation="I use Python",
            source_id="conversation-1",
        )
    )

    assert len(result["validated_memories"]) == 1
    assert len(result["written_memories"]) == 1

    written_memory = result["written_memories"][0]

    assert written_memory.memory_id == "memory-1"
    assert written_memory.value == "Python"

    stored_memory = get(
        "memory-1",
        db_path,
    )

    assert stored_memory is not None
    assert stored_memory.memory_id == "memory-1"
    assert stored_memory.value == "Python"


def test_invalid_memory_is_not_written_to_sqlite(tmp_path):
    db_path = tmp_path / "workflow.db"

    initialize_database(db_path)

    extractor = StaticMemoryExtractor(
        MemoryExtractionResult(
            memories=[
                make_candidate(
                    confidence=0.49,
                ),
            ]
        )
    )

    workflow = build_memory_workflow(
        extractor,
        db_path=db_path,
        id_factory=lambda: "memory-invalid",
    )

    result = workflow.invoke(
        MemoryAgentState(
            conversation="I use Python",
            source_id="conversation-1",
        )
    )

    assert len(result["extracted_memories"]) == 1
    assert result["validated_memories"] == []
    assert result["written_memories"] == []

    stored_memory = get(
        "memory-invalid",
        db_path,
    )

    assert stored_memory is None


def test_multiple_valid_memories_are_written(tmp_path):
    db_path = tmp_path / "workflow.db"

    initialize_database(db_path)

    candidates = [
        make_candidate(
            value="Python",
            confidence=0.95,
        ),
        make_candidate(
            value="Java",
            confidence=0.9,
        ),
    ]

    extractor = StaticMemoryExtractor(
        MemoryExtractionResult(
            memories=candidates,
        )
    )

    ids = iter(
        [
            "memory-1",
            "memory-2",
        ]
    )

    workflow = build_memory_workflow(
        extractor,
        db_path=db_path,
        id_factory=lambda: next(ids),
    )

    result = workflow.invoke(
        MemoryAgentState(
            conversation="I use Python and Java",
            source_id="conversation-1",
        )
    )

    assert len(result["validated_memories"]) == 2
    assert len(result["written_memories"]) == 2

    memory_1 = get(
        "memory-1",
        db_path,
    )

    memory_2 = get(
        "memory-2",
        db_path,
    )

    assert memory_1 is not None
    assert memory_2 is not None

    assert memory_1.value == "Python"
    assert memory_2.value == "Java"