"""Bootstrap confidence intervals over questions.

The question is the unit of resampling: a model's answers to one question
on both warehouses move together, so resampling rows would understate the
uncertainty. A fixed seed makes every interval reproducible.
"""

from dataclasses import dataclass

import numpy as np

SEED = 20261112
RESAMPLES = 10_000
LEVEL = 0.95


@dataclass(frozen=True)
class Interval:
    estimate: float
    low: float
    high: float

    @property
    def excludes_zero(self) -> bool:
        return self.low > 0 or self.high < 0


def percentile_interval(estimate: float, samples: np.ndarray) -> Interval:
    tail = (1 - LEVEL) / 2 * 100
    low, high = np.percentile(samples, [tail, 100 - tail])
    return Interval(estimate, float(low), float(high))


def mean_interval(values: list[float], seed: int = SEED) -> Interval:
    """CI of the mean of per-question values (1.0 correct, 0.0 not, or a share)."""
    data = np.asarray(values, dtype=float)
    if data.size == 0:
        raise ValueError("no values to resample")
    generator = np.random.default_rng(seed)
    picks = generator.integers(0, data.size, size=(RESAMPLES, data.size))
    return percentile_interval(float(data.mean()), data[picks].mean(axis=1))


def paired_difference_interval(
    first: list[float], second: list[float], seed: int = SEED
) -> Interval:
    """CI of mean(first) - mean(second) where both are measured on the same questions."""
    if len(first) != len(second):
        raise ValueError("paired values need the same questions")
    differences = np.asarray(first, dtype=float) - np.asarray(second, dtype=float)
    return mean_interval(list(differences), seed)


def unpaired_difference_interval(
    first: list[float], second: list[float], seed: int = SEED
) -> Interval:
    """CI of mean(first) - mean(second) for two separate sets of questions."""
    first_data = np.asarray(first, dtype=float)
    second_data = np.asarray(second, dtype=float)
    if first_data.size == 0 or second_data.size == 0:
        raise ValueError("no values to resample")
    generator = np.random.default_rng(seed)
    first_picks = generator.integers(0, first_data.size, size=(RESAMPLES, first_data.size))
    second_picks = generator.integers(0, second_data.size, size=(RESAMPLES, second_data.size))
    samples = first_data[first_picks].mean(axis=1) - second_data[second_picks].mean(axis=1)
    estimate = float(first_data.mean() - second_data.mean())
    return percentile_interval(estimate, samples)
