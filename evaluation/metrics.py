from dataclasses import dataclass


@dataclass(frozen=True)
class EvaluationMetrics:
    total: int
    correct: int

    @property
    def accuracy(self) -> float:
        if self.total == 0:
            return 0.0

        return self.correct / self.total


@dataclass(frozen=True)
class EvaluationComparison:
    temporal: EvaluationMetrics
    naive: EvaluationMetrics

    @property
    def accuracy_difference(self) -> float:
        return (
            self.temporal.accuracy
            - self.naive.accuracy
        )


def calculate_accuracy(
    expected: list[str],
    actual: list[str],
) -> EvaluationMetrics:
    if len(expected) != len(actual):
        raise ValueError(
            "expected and actual must have equal length"
        )

    correct = sum(
        1
        for expected_value, actual_value
        in zip(
            expected,
            actual,
        )
        if expected_value == actual_value
    )

    return EvaluationMetrics(
        total=len(expected),
        correct=correct,
    )


def compare_evaluations(
    temporal_expected: list[str],
    temporal_actual: list[str],
    naive_actual: list[str],
) -> EvaluationComparison:
    if len(temporal_expected) != len(
        temporal_actual
    ):
        raise ValueError(
            "temporal expected and actual "
            "must have equal length"
        )

    if len(temporal_expected) != len(
        naive_actual
    ):
        raise ValueError(
            "expected and naive actual "
            "must have equal length"
        )

    temporal_metrics = calculate_accuracy(
        temporal_expected,
        temporal_actual,
    )

    naive_metrics = calculate_accuracy(
        temporal_expected,
        naive_actual,
    )

    return EvaluationComparison(
        temporal=temporal_metrics,
        naive=naive_metrics,
    )