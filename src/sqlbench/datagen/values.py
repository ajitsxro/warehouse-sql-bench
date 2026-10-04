"""Draw one column's values from its generator hints.

Each column gets its own seed, so adding or reordering columns never changes
the values of another column. Vectorized draws use numpy. Nested values are
built one at a time with Python's random module, which is much faster than
numpy for many small scalar draws.
"""

import datetime
import hashlib
import random
import string
from decimal import Decimal
from typing import Any

import numpy as np
from faker import Faker

from sqlbench.schema.spec import ColumnSpec, GenSpec, JsonShape
from sqlbench.schema.types import INTEGER_KINDS, NUMERIC_KINDS, LogicalType, TypeKind

FAKER_LOCALE = "en_US"
DEFAULT_FAKER_POOL = 5000
# Self-references pick a parent among the first 1/SPAN_OF_CONTROL of earlier
# rows, which keeps an org chart a handful of levels deep.
SPAN_OF_CONTROL = 6


class JsonNull:
    """A VARIANT whose value is JSON null, as opposed to SQL NULL (None)."""


JSON_NULL = JsonNull()


def derive_seed(*parts: object) -> int:
    """Stable 64-bit seed from names; Python's hash() changes between runs."""
    text = ":".join(str(part) for part in parts)
    digest = hashlib.sha256(text.encode()).digest()
    return int.from_bytes(digest[:8], "big")


def generate_values(
    column: ColumnSpec, row_count: int, seed: int, parent_keys: list[Any] | None = None
) -> list[Any]:
    """Values for a column that does not depend on another column in its row."""
    rng = np.random.default_rng(seed)
    gen = column.gen
    kind = column.logical_type.kind

    if column.references is not None:
        assert parent_keys is not None
        values = pick_parent_keys(gen, parent_keys, row_count, rng)
    elif gen.values is not None:
        values = pick_choices(gen, row_count, rng)
    elif kind in NUMERIC_KINDS:
        values = draw_numbers(gen, column.logical_type, row_count, rng)
    elif kind == TypeKind.STRING:
        values = draw_strings(gen, row_count, seed, rng)
    elif kind == TypeKind.BOOL:
        values = [bool(flag) for flag in rng.random(row_count) < 0.5]
    elif kind in (TypeKind.DATE, TypeKind.TIMESTAMP_NTZ, TypeKind.TIMESTAMP_TZ):
        values = draw_temporal(gen, kind, row_count, rng)
    else:
        assert gen.shape is not None
        shape_rng = random.Random(seed)
        values = [generate_shape_value(gen.shape, shape_rng) for _ in range(row_count)]

    return apply_null_rate(values, gen.null_rate, rng)


def generate_after(column: ColumnSpec, anchors: list[Any], seed: int) -> list[Any]:
    """Values later than the anchor column in the same row, by up to max_gap_days."""
    rng = np.random.default_rng(seed)
    gen = column.gen
    assert gen.max_gap_days is not None
    values: list[Any] = []
    if column.logical_type.kind == TypeKind.DATE:
        gap_days = rng.integers(0, int(gen.max_gap_days) + 1, len(anchors))
        for anchor, days in zip(anchors, gap_days, strict=True):
            values.append(None if anchor is None else anchor + datetime.timedelta(days=int(days)))
    else:
        gap_seconds = rng.integers(0, int(gen.max_gap_days * 86400) + 1, len(anchors))
        for anchor, seconds in zip(anchors, gap_seconds, strict=True):
            gap = datetime.timedelta(seconds=int(seconds))
            values.append(None if anchor is None else anchor + gap)
    return apply_null_rate(values, gen.null_rate, rng)


def generate_self_references(column: ColumnSpec, keys: list[Any], seed: int) -> list[Any]:
    """Parent keys for a hierarchy. The first row is the root and gets NULL.

    Every other row points at an earlier row, so following parents always ends
    at a root and the hierarchy has no cycles.
    """
    rng = np.random.default_rng(seed)
    draws = rng.random(len(keys))
    parents: list[Any] = [None]
    for index in range(1, len(keys)):
        candidates = max(1, index // SPAN_OF_CONTROL)
        parents.append(keys[int(draws[index] * candidates)])
    return apply_null_rate(parents, column.gen.null_rate, rng)


def apply_null_rate(values: list[Any], null_rate: float, rng: np.random.Generator) -> list[Any]:
    if null_rate == 0:
        return values
    is_null = rng.random(len(values)) < null_rate
    for index in np.flatnonzero(is_null):
        values[index] = None
    return values


def pick_parent_keys(
    gen: GenSpec, parent_keys: list[Any], row_count: int, rng: np.random.Generator
) -> list[Any]:
    if gen.distribution == "uniform":
        indexes = rng.integers(0, len(parent_keys), row_count)
    else:
        # Zipf over a shuffled ranking, so the heavy parents are not just the lowest ids.
        ranking = rng.permutation(len(parent_keys))
        weights = 1.0 / np.arange(1, len(parent_keys) + 1) ** gen.exponent
        indexes = ranking[rng.choice(len(parent_keys), row_count, p=weights / weights.sum())]
    return [parent_keys[index] for index in indexes]


def pick_choices(gen: GenSpec, row_count: int, rng: np.random.Generator) -> list[Any]:
    assert gen.values is not None
    if gen.distribution == "cycle":
        return [gen.values[index % len(gen.values)] for index in range(row_count)]
    probabilities = None
    if gen.weights is not None:
        weights = np.array(gen.weights, dtype=float)
        probabilities = weights / weights.sum()
    indexes = rng.choice(len(gen.values), row_count, p=probabilities)
    return [gen.values[index] for index in indexes]


def draw_numbers(
    gen: GenSpec, logical_type: LogicalType, row_count: int, rng: np.random.Generator
) -> list[Any]:
    kind = logical_type.kind
    low = None if gen.min is None else as_number(gen.min)
    high = None if gen.max is None else as_number(gen.max)

    raw: np.ndarray[Any, Any]
    if gen.distribution == "sequence":
        start = 1 if low is None else int(low)
        return list(range(start, start + row_count))
    if gen.distribution == "uniform":
        assert low is not None and high is not None
        if kind in INTEGER_KINDS:
            integers: list[Any] = rng.integers(int(low), int(high) + 1, row_count).tolist()
            return integers
        raw = rng.uniform(low, high, row_count)
    elif gen.distribution == "zipf":
        assert low is not None and high is not None
        raw = low + rng.zipf(gen.exponent, row_count) - 1
    elif gen.distribution == "normal":
        assert gen.mean is not None and gen.sigma is not None
        raw = rng.normal(gen.mean, gen.sigma, row_count)
    else:
        assert gen.mean is not None and gen.sigma is not None
        raw = rng.lognormal(gen.mean, gen.sigma, row_count)

    if kind == TypeKind.DECIMAL:
        assert logical_type.precision is not None and logical_type.scale is not None
        # Keep values representable in decimal(p,s) even when max is not given.
        largest = 10 ** (logical_type.precision - logical_type.scale) - 10**-logical_type.scale
        high = largest if high is None else min(high, largest)
    clipped = np.clip(raw, low, high)

    if kind in INTEGER_KINDS:
        rounded: list[Any] = np.rint(clipped).astype(np.int64).tolist()
        return rounded
    if kind == TypeKind.DECIMAL:
        assert logical_type.scale is not None
        scale = logical_type.scale
        scaled = np.rint(clipped * 10**scale).astype(np.int64)
        return [Decimal(int(value)).scaleb(-scale) for value in scaled]
    floats: list[Any] = clipped.tolist()
    return floats


def draw_strings(gen: GenSpec, row_count: int, seed: int, rng: np.random.Generator) -> list[Any]:
    if gen.pattern is not None:
        pattern_rng = random.Random(seed)
        return [fill_pattern(gen.pattern, pattern_rng) for _ in range(row_count)]

    assert gen.faker is not None
    faker = Faker(FAKER_LOCALE)
    faker.seed_instance(seed)
    provider = getattr(faker, gen.faker, None)
    if provider is None:
        raise ValueError(f"unknown faker provider {gen.faker!r}")
    # Faker is slow, so draw a fixed pool once and sample rows from it.
    pool_size = min(gen.cardinality or DEFAULT_FAKER_POOL, row_count)
    pool = [str(provider()) for _ in range(pool_size)]
    return [pool[index] for index in rng.integers(0, pool_size, row_count)]


def draw_temporal(
    gen: GenSpec, kind: TypeKind, row_count: int, rng: np.random.Generator
) -> list[Any]:
    assert gen.min is not None and gen.max is not None
    if kind == TypeKind.DATE:
        first_day = as_date(gen.min)
        span_days = (as_date(gen.max) - first_day).days
        offsets = rng.integers(0, span_days + 1, row_count)
        return [first_day + datetime.timedelta(days=int(days)) for days in offsets]

    tzinfo = datetime.UTC if kind == TypeKind.TIMESTAMP_TZ else None
    start = as_datetime(gen.min).replace(tzinfo=tzinfo)
    span_seconds = int((as_datetime(gen.max).replace(tzinfo=tzinfo) - start).total_seconds())
    offsets = rng.integers(0, span_seconds + 1, row_count)
    return [start + datetime.timedelta(seconds=int(seconds)) for seconds in offsets]


def as_number(value: object) -> float:
    if isinstance(value, int | float):
        return float(value)
    raise ValueError(f"expected a number, got {value!r}")


def as_date(value: object) -> datetime.date:
    if isinstance(value, datetime.datetime):
        return value.date()
    if isinstance(value, datetime.date):
        return value
    raise ValueError(f"expected a date, got {value!r}")


def as_datetime(value: object) -> datetime.datetime:
    if isinstance(value, datetime.datetime):
        return value.replace(tzinfo=None)
    if isinstance(value, datetime.date):
        return datetime.datetime(value.year, value.month, value.day)
    raise ValueError(f"expected a date or timestamp, got {value!r}")


def fill_pattern(pattern: str, rng: random.Random) -> str:
    """Replace each # with a digit and each ? with an uppercase letter."""
    characters: list[str] = []
    for character in pattern:
        if character == "#":
            characters.append(rng.choice(string.digits))
        elif character == "?":
            characters.append(rng.choice(string.ascii_uppercase))
        else:
            characters.append(character)
    return "".join(characters)


def generate_shape_value(shape: JsonShape, rng: random.Random) -> Any:
    """One nested value. Objects come back as dicts with optional keys left out."""
    if shape.null_rate > 0 and rng.random() < shape.null_rate:
        return None
    if shape.type == "object":
        value: dict[str, Any] = {}
        for name, field_shape in shape.fields.items():
            if field_shape.optional > 0 and rng.random() < field_shape.optional:
                continue
            value[name] = generate_shape_value(field_shape, rng)
        return value
    if shape.type == "array":
        assert shape.element is not None
        length = rng.randint(shape.min_items, shape.max_items)
        return [generate_shape_value(shape.element, rng) for _ in range(length)]
    if shape.type == "int":
        return rng.randint(int(shape.min), int(shape.max))
    if shape.type == "number":
        return round(rng.uniform(shape.min, shape.max), shape.scale)
    if shape.type == "bool":
        return rng.random() < 0.5
    if shape.values is not None:
        return rng.choice(shape.values)
    assert shape.pattern is not None
    return fill_pattern(shape.pattern, rng)
