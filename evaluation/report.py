from pathlib import Path

from evaluation.runner import (
    EvaluationResult,
    run_evaluation,
)


def generate_report(
    db_path: Path,
):
    result = run_evaluation(
        db_path
    )

    temporal = (
        result.comparison.temporal
    )

    naive = (
        result.comparison.naive
    )

    return {
        "total_cases": temporal.total,
        "temporal": {
            "correct": temporal.correct,
            "accuracy": temporal.accuracy,
        },
        "naive": {
            "correct": naive.correct,
            "accuracy": naive.accuracy,
        },
        "accuracy_difference": (
            result.comparison
            .accuracy_difference
        ),
        "temporal_values": (
            result.temporal_values
        ),
        "naive_values": (
            result.naive_values
        ),
    }


if __name__ == "__main__":
    report = generate_report(
        Path("data/memory.db")
    )

    print(
        "Temporal Memory Evaluation"
    )

    print(
        f"Evaluation cases: "
        f"{report['total_cases']}"
    )

    print()

    print(
        "Temporal memory:"
    )

    print(
        f"  Correct: "
        f"{report['temporal']['correct']}"
    )

    print(
        f"  Accuracy: "
        f"{report['temporal']['accuracy']:.3f}"
    )

    print()

    print(
        "Naive memory:"
    )

    print(
        f"  Correct: "
        f"{report['naive']['correct']}"
    )

    print(
        f"  Accuracy: "
        f"{report['naive']['accuracy']:.3f}"
    )

    print()

    print(
        "Accuracy difference: "
        f"{report['accuracy_difference']:.3f}"
    )

    print()

    print(
        "Temporal predictions:"
    )

    for value in report[
        "temporal_values"
    ]:
        print(
            f"  {value}"
        )

    print()

    print(
        "Naive predictions:"
    )

    for value in report[
        "naive_values"
    ]:
        print(
            f"  {value}"
        )