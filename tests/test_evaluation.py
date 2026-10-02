from evaluation.dataset import (
    DATASET,
)
from evaluation.metrics import (
    calculate_accuracy,
    compare_evaluations,
)


def test_dataset_uses_timezone_aware_datetimes():
    assert len(DATASET) == 3

    for case in DATASET:
        assert (
            case.query_time.tzinfo
            is not None
        )


def test_calculate_accuracy():
    metrics = calculate_accuracy(
        [
            "Node.js",
            "Go",
            "Python",
        ],
        [
            "Node.js",
            "Go",
            "Python",
        ],
    )

    assert metrics.total == 3
    assert metrics.correct == 3
    assert metrics.accuracy == 1.0


def test_calculate_accuracy_with_failure():
    metrics = calculate_accuracy(
        [
            "Node.js",
            "Go",
            "Python",
        ],
        [
            "Python",
            "Python",
            "Python",
        ],
    )

    assert metrics.total == 3
    assert metrics.correct == 1
    assert (
        metrics.accuracy
        == 1 / 3
    )


def test_calculate_accuracy_rejects_different_lengths():
    try:
        calculate_accuracy(
            ["Python"],
            [],
        )
    except ValueError as exc:
        assert (
            str(exc)
            == "expected and actual must have equal length"
        )
    else:
        raise AssertionError(
            "ValueError was not raised"
        )


def test_compare_evaluations():
    comparison = compare_evaluations(
        [
            "Node.js",
            "Go",
            "Python",
        ],
        [
            "Node.js",
            "Go",
            "Python",
        ],
        [
            "Python",
            "Python",
            "Python",
        ],
    )

    assert (
        comparison.temporal.accuracy
        == 1.0
    )

    assert (
        comparison.naive.accuracy
        == 1 / 3
    )

    assert abs(
        comparison.accuracy_difference
        - (2 / 3)
    ) < 1e-9


def test_compare_evaluations_rejects_temporal_length_mismatch():
    try:
        compare_evaluations(
            [
                "Python",
            ],
            [],
            [
                "Python",
            ],
        )
    except ValueError as exc:
        assert (
            str(exc)
            == (
                "temporal expected and actual "
                "must have equal length"
            )
        )
    else:
        raise AssertionError(
            "ValueError was not raised"
        )


def test_compare_evaluations_rejects_naive_length_mismatch():
    try:
        compare_evaluations(
            [
                "Python",
            ],
            [
                "Python",
            ],
            [],
        )
    except ValueError as exc:
        assert (
            str(exc)
            == (
                "expected and naive actual "
                "must have equal length"
            )
        )
    else:
        raise AssertionError(
            "ValueError was not raised"
        )