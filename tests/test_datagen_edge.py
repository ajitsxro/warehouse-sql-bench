from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from sqlbench.datagen.edge import BOUNDARY_DATES, BOUNDARY_TIMESTAMPS, DST_INSTANTS
from sqlbench.datagen.generator import find_integrity_problems, generate_tables, to_arrow_table
from sqlbench.datagen.tables import Profile, TableData
from sqlbench.datagen.values import JSON_NULL
from sqlbench.schema.spec import load_schema

SCHEMA = load_schema(Path(__file__).parent / "fixtures" / "mini_shop.yaml")


@pytest.fixture(scope="module")
def edge() -> dict[str, TableData]:
    return generate_tables(SCHEMA, Profile.EDGE, seed=5)


def present(values: list[Any]) -> list[Any]:
    return [value for value in values if value is not None]


def test_edge_data_keeps_keys_and_foreign_keys_valid(edge: dict[str, TableData]) -> None:
    assert find_integrity_problems(SCHEMA, edge) == []


def test_main_profile_gets_no_injectors() -> None:
    main = generate_tables(SCHEMA, Profile.MAIN, seed=5)

    assert all(table.injectors == [] for table in main.values())


def test_injectors_are_recorded_per_table(edge: dict[str, TableData]) -> None:
    assert edge["customers"].injectors == [
        "empty_groups",
        "nulls",
        "boundary_dates",
        "semi_structured",
        "messy_strings",
    ]
    assert "empty_groups" not in edge["events"].injectors
    assert "duplicates" in edge["events"].injectors
    assert "duplicates" not in edge["orders"].injectors
    assert "ties" in edge["orders"].injectors


def test_some_customers_have_no_orders_and_no_events(edge: dict[str, TableData]) -> None:
    customers = set(edge["customers"].columns["customer_id"])
    used = set(edge["orders"].columns["customer_id"]) | set(edge["events"].columns["customer_id"])

    assert len(customers - used) >= 1


def test_every_nullable_column_has_nulls(edge: dict[str, TableData]) -> None:
    for table in SCHEMA.tables:
        for column in table.columns:
            if column.nullable:
                assert None in edge[table.name].columns[column.name], column.name


def test_tables_without_primary_key_get_exact_duplicate_rows(edge: dict[str, TableData]) -> None:
    rows = to_arrow_table(edge["events"]).to_pylist()
    row_counts = Counter(tuple(sorted(row.items())) for row in rows)

    assert max(row_counts.values()) >= 2


@pytest.mark.parametrize(("table", "column"), [("orders", "total"), ("orders", "placed_at")])
def test_ties_at_rank_one_and_at_the_top_ten_cut(
    edge: dict[str, TableData], table: str, column: str
) -> None:
    ranked = sorted(present(edge[table].columns[column]), reverse=True)

    assert ranked[0] == ranked[1]
    assert ranked[9] == ranked[10]


def test_boundary_dates_and_dst_instants_are_present(edge: dict[str, TableData]) -> None:
    signup_dates = set(edge["customers"].columns["signup_date"])
    placed = set(edge["orders"].columns["placed_at"])
    event_times = set(edge["events"].columns["event_ts"])

    assert set(BOUNDARY_DATES) <= signup_dates
    assert set(BOUNDARY_TIMESTAMPS) <= placed
    assert set(DST_INSTANTS) <= event_times


def test_variant_edge_cases(edge: dict[str, TableData]) -> None:
    attributes = edge["orders"].columns["attributes"]
    objects = [value for value in attributes if isinstance(value, dict)]

    assert None in attributes
    assert JSON_NULL in attributes
    assert any("channel" not in value for value in objects)
    assert any("channel" in value and value["channel"] is None for value in objects)
    assert any(value.get("items") == [] for value in objects)
    item_lists = [value["items"] for value in objects if isinstance(value.get("items"), list)]
    assert any(len(items) == 1 for items in item_lists)


def test_typed_nested_edge_cases(edge: dict[str, TableData]) -> None:
    tags = edge["customers"].columns["tags"]
    addresses = edge["customers"].columns["address"]
    specs = edge["orders"].columns["specs"]

    assert [] in tags
    assert any(isinstance(value, list) and len(value) == 1 for value in tags)
    assert any(address["city"] is None for address in addresses)
    assert {} in specs


def test_strings_get_padding_and_case_changes(edge: dict[str, TableData]) -> None:
    segments = present(edge["customers"].columns["segment"])

    assert any(value != value.strip() for value in segments)
    assert any(value.isupper() for value in segments)
    assert {value.strip().lower() for value in segments} == {"consumer", "business"}
