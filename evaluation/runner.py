from dataclasses import dataclass
from pathlib import Path

from app.agent.query_resolver import (
    MemoryQuery,
    StaticQueryResolver,
)
from app.agent.reasoner import StaticAgentReasoner
from app.agent.state import MemoryAgentState
from app.agent.workflow import (
    build_memory_query_workflow,
)
from app.storage.memory_repository import (
    list_by_key,
)
from evaluation.dataset import (
    DATASET,
    EvaluationCase,
)
from evaluation.metrics import (
    EvaluationComparison,
    EvaluationMetrics,
    calculate_accuracy,
    compare_evaluations,
)


@dataclass(frozen=True)
class EvaluationResult:
    comparison: EvaluationComparison
    temporal_values: list[str]
    naive_values: list[str]


def run_temporal_evaluation(
    db_path: Path,
) -> EvaluationMetrics:
    expected = []
    actual = []

    for case in DATASET:
        resolver = StaticQueryResolver(
            MemoryQuery(
                memory_keys=[
                    case.memory_key
                ],
                query_time=case.query_time,
            )
        )

        workflow = build_memory_query_workflow(
            resolver,
            reasoner=StaticAgentReasoner(),
            db_path=db_path,
        )

        result = workflow.invoke(
            MemoryAgentState(
                conversation=case.query,
                source_id="evaluation",
            )
        )

        memories = result[
            "context_memories"
        ]

        actual_value = (
            memories[0].value
            if memories
            else ""
        )

        expected.append(
            case.expected_value
        )

        actual.append(
            actual_value
        )

    return calculate_accuracy(
        expected,
        actual,
    )


def _latest_memory_value(
    case: EvaluationCase,
    db_path: Path,
) -> str:
    memories = list_by_key(
        case.memory_key,
        db_path,
    )

    if not memories:
        return ""

    latest = max(
        memories,
        key=lambda memory: memory.valid_from,
    )

    return latest.value


def run_naive_evaluation(
    db_path: Path,
) -> EvaluationMetrics:
    expected = []
    actual = []

    for case in DATASET:
        expected.append(
            case.expected_value
        )

        actual.append(
            _latest_memory_value(
                case,
                db_path,
            )
        )

    return calculate_accuracy(
        expected,
        actual,
    )


def run_evaluation(
    db_path: Path,
) -> EvaluationResult:
    expected = [
        case.expected_value
        for case in DATASET
    ]

    temporal_values = []

    for case in DATASET:
        resolver = StaticQueryResolver(
            MemoryQuery(
                memory_keys=[
                    case.memory_key
                ],
                query_time=case.query_time,
            )
        )

        workflow = build_memory_query_workflow(
            resolver,
            reasoner=StaticAgentReasoner(),
            db_path=db_path,
        )

        result = workflow.invoke(
            MemoryAgentState(
                conversation=case.query,
                source_id="evaluation",
            )
        )

        memories = result[
            "context_memories"
        ]

        temporal_values.append(
            memories[0].value
            if memories
            else ""
        )

    naive_values = [
        _latest_memory_value(
            case,
            db_path,
        )
        for case in DATASET
    ]

    comparison = compare_evaluations(
        expected,
        temporal_values,
        naive_values,
    )

    return EvaluationResult(
        comparison=comparison,
        temporal_values=temporal_values,
        naive_values=naive_values,
    )