"""Bootstrap intervals."""

import pytest

from sqlbench.analysis.stats import (
    mean_interval,
    paired_difference_interval,
    unpaired_difference_interval,
)


def test_all_correct_has_a_zero_width_interval() -> None:
    interval = mean_interval([1.0] * 40)

    assert (interval.estimate, interval.low, interval.high) == (1.0, 1.0, 1.0)


def test_interval_contains_the_estimate_and_is_reproducible() -> None:
    values = [1.0] * 30 + [0.0] * 10

    interval = mean_interval(values)

    assert interval.estimate == 0.75
    assert interval.low < 0.75 < interval.high
    assert interval == mean_interval(values)


def test_more_questions_give_a_narrower_interval() -> None:
    small = mean_interval([1.0] * 6 + [0.0] * 2)
    large = mean_interval([1.0] * 60 + [0.0] * 20)

    assert large.high - large.low < small.high - small.low


def test_paired_difference_sees_a_consistent_gap() -> None:
    first = [1.0] * 40
    second = [1.0] * 30 + [0.0] * 10

    interval = paired_difference_interval(first, second)

    assert interval.estimate == pytest.approx(0.25)
    assert interval.excludes_zero


def test_a_one_question_difference_is_not_significant() -> None:
    first = [1.0] * 40
    second = [1.0] * 39 + [0.0]

    assert not paired_difference_interval(first, second).excludes_zero


def test_paired_values_need_the_same_questions() -> None:
    with pytest.raises(ValueError, match="same questions"):
        paired_difference_interval([1.0], [1.0, 0.0])


def test_unpaired_difference_of_separate_groups() -> None:
    interval = unpaired_difference_interval([1.0] * 50, [0.0] * 50)

    assert (interval.estimate, interval.low, interval.high) == (1.0, 1.0, 1.0)
