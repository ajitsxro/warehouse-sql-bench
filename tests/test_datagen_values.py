import datetime
import math
import random
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from sqlbench.datagen.values import (
    derive_seed,
    fill_pattern,
    generate_after,
    generate_shape_value,
    generate_values,
)
from sqlbench.schema.spec import ColumnSpec, JsonShape, load_schema

SCHEMA = load_schema(Path(__file__).parent / "fixtures" / "mini_shop.yaml")
ORDERS = SCHEMA.table("orders")


def column(spec: dict[str, Any]) -> ColumnSpec:
    return ColumnSpec.model_validate(spec)


def within_binomial_tolerance(observed: int, total: int, rate: float) -> bool:
    # Four standard deviations: a correct generator fails about once in 15,000 runs.
    tolerance = 4 * math.sqrt(total * rate * (1 - rate))
    return abs(observed - total * rate) <= tolerance


def test_derive_seed_is_stable_and_sensitive_to_every_part() -> None:
    assert derive_seed(7, "main", "orders", "total") == derive_seed(7, "main", "orders", "total")
    assert derive_seed(7, "main", "orders", "total") != derive_seed(7, "edge", "orders", "total")
    assert derive_seed(7, "main", "orders", "total") != derive_seed(8, "main", "orders", "total")
    # sha256-based, so the value never changes between Python runs or machines.
    assert derive_seed("a") == 0xCA978112CA1BBDCA


@pytest.mark.parametrize("name", [c.name for c in ORDERS.columns if c.gen.after is None])
def test_same_seed_gives_same_values(name: str) -> None:
    target = ORDERS.column(name)
    parents = list(range(1, 51)) if target.references else None

    first = generate_values(target, 300, seed=11, parent_keys=parents)
    second = generate_values(target, 300, seed=11, parent_keys=parents)
    other = generate_values(target, 300, seed=12, parent_keys=parents)

    assert first == second
    if target.gen.distribution != "sequence":
        assert first != other


def test_sequence_counts_up_from_min() -> None:
    values = generate_values(
        column({"name": "id", "type": "bigint", "gen": {"distribution": "sequence", "min": 100}}),
        5,
        seed=1,
    )
    assert values == [100, 101, 102, 103, 104]


def test_cycle_assigns_values_in_order() -> None:
    target = column(
        {"name": "tier", "type": "string", "gen": {"distribution": "cycle", "values": ["a", "b"]}}
    )

    assert generate_values(target, 5, seed=1) == ["a", "b", "a", "b", "a"]


def test_uniform_ints_cover_both_inclusive_bounds() -> None:
    values = generate_values(ORDERS.column("quantity"), 2000, seed=1)

    assert set(values) == {1, 2, 3, 4, 5}


def test_decimals_keep_scale_and_stay_within_bounds() -> None:
    values = generate_values(ORDERS.column("total"), 5000, seed=1)

    assert all(isinstance(value, Decimal) for value in values)
    assert all(value.as_tuple().exponent == -2 for value in values)
    assert min(values) >= Decimal("1.00")
    assert max(values) <= Decimal("2000.00")


def test_decimal_without_max_is_clipped_to_the_type_range() -> None:
    target = column(
        {
            "name": "amount",
            "type": "decimal(4,2)",
            "gen": {"distribution": "lognormal", "mean": 6, "sigma": 1},
        }
    )
    values = generate_values(target, 1000, seed=1)

    assert max(values) == Decimal("99.99")


def test_normal_is_clipped_at_min() -> None:
    target = column(
        {
            "name": "w",
            "type": "double",
            "gen": {"distribution": "normal", "mean": 0, "sigma": 1, "min": 0},
        }
    )
    values = generate_values(target, 1000, seed=1)

    assert min(values) == 0.0


def test_weighted_choices_follow_weights() -> None:
    segment = SCHEMA.table("customers").column("segment")
    values = generate_values(segment, 20000, seed=1)
    non_null = [value for value in values if value is not None]

    consumer_count = non_null.count("consumer")
    assert within_binomial_tolerance(consumer_count, len(non_null), 0.75)


def test_null_rate_is_respected() -> None:
    values = generate_values(SCHEMA.table("customers").column("segment"), 20000, seed=1)

    assert within_binomial_tolerance(values.count(None), 20000, 0.05)


def test_zipf_foreign_keys_are_skewed_and_only_use_parent_keys() -> None:
    target = column(
        {
            "name": "customer_id",
            "type": "bigint",
            "references": "customers.customer_id",
            "gen": {"distribution": "zipf"},
        }
    )
    parents = list(range(1, 101))
    values = generate_values(target, 10000, seed=1, parent_keys=parents)

    assert set(values) <= set(parents)
    busiest_share = max(values.count(key) for key in parents) / len(values)
    assert busiest_share > 0.1  # a uniform pick would give about 0.01


def test_timestamps_respect_bounds_and_timezone_kind() -> None:
    ntz = generate_values(ORDERS.column("placed_at"), 1000, seed=1)
    tz = generate_values(SCHEMA.table("events").column("event_ts"), 1000, seed=1)

    assert all(value.tzinfo is None for value in ntz)
    assert all(value.tzinfo == datetime.UTC for value in tz)
    assert min(ntz) >= datetime.datetime(2023, 1, 1)
    assert max(ntz) <= datetime.datetime(2024, 12, 31)


def test_after_values_follow_their_anchor_within_the_gap() -> None:
    anchors: list[Any] = [datetime.datetime(2024, 1, 1), None] * 500
    shipped = generate_after(ORDERS.column("shipped_at"), anchors, seed=1)

    for anchor, value in zip(anchors, shipped, strict=True):
        if anchor is None:
            assert value is None
        elif value is not None:
            assert anchor <= value <= anchor + datetime.timedelta(days=10)


def test_faker_pool_limits_cardinality() -> None:
    values = generate_values(SCHEMA.table("customers").column("full_name"), 5000, seed=1)

    assert len(set(values)) <= 200


def test_unknown_faker_provider_is_an_error() -> None:
    target = column({"name": "x", "type": "string", "gen": {"faker": "not_a_provider"}})

    with pytest.raises(ValueError, match="unknown faker provider"):
        generate_values(target, 5, seed=1)


def test_fill_pattern() -> None:
    value = fill_pattern("SKU-##-??", random.Random(1))

    assert value[:4] == "SKU-"
    assert value[4:6].isdigit()
    assert value[7:].isalpha() and value[7:].isupper()


def test_shapes_leave_out_optional_keys_and_respect_array_lengths() -> None:
    shape = ORDERS.column("attributes").gen.shape
    assert shape is not None
    rng = random.Random(1)
    values = [generate_shape_value(shape, rng) for _ in range(2000)]

    assert all("channel" in value for value in values)
    assert any("coupon" not in value for value in values)
    assert any("coupon" in value for value in values)
    assert all(1 <= len(value["items"]) <= 4 for value in values)
    gifts = [value["gift"] for value in values if "gift" in value]
    assert None in gifts and True in gifts


def test_shape_null_rate_gives_json_null() -> None:
    shape = JsonShape.model_validate({"type": "int", "null_rate": 1.0})

    assert generate_shape_value(shape, random.Random(1)) is None
