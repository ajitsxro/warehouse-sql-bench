import copy
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from sqlbench.schema.spec import TableSpec

FIXTURE = Path(__file__).parent / "fixtures" / "mini_shop.yaml"


def fixture_dict() -> dict[str, Any]:
    with FIXTURE.open() as handle:
        loaded: dict[str, Any] = yaml.safe_load(handle)
    return copy.deepcopy(loaded)


def find_column(raw: dict[str, Any], table: str, column: str) -> dict[str, Any]:
    for table_raw in raw["tables"]:
        if table_raw["name"] == table:
            for column_raw in table_raw["columns"]:
                if column_raw["name"] == column:
                    found: dict[str, Any] = column_raw
                    return found
    raise KeyError(f"{table}.{column}")


def null_rate_without_nullable(raw: dict[str, Any]) -> None:
    find_column(raw, "customers", "full_name")["gen"]["null_rate"] = 0.1


def string_without_generator(raw: dict[str, Any]) -> None:
    find_column(raw, "events", "event_name")["gen"] = {}


def struct_shape_missing_field(raw: dict[str, Any]) -> None:
    del find_column(raw, "customers", "address")["gen"]["shape"]["fields"]["zip"]


def array_shape_wrong_element(raw: dict[str, Any]) -> None:
    column = find_column(raw, "customers", "tags")
    column["gen"]["shape"]["element"] = {"type": "int"}


def edge_rows_too_many(raw: dict[str, Any]) -> None:
    raw["tables"][1]["edge_rows"] = 5000


def after_wrong_type(raw: dict[str, Any]) -> None:
    find_column(raw, "orders", "shipped_at")["gen"]["after"] = "total"


def nullable_primary_key(raw: dict[str, Any]) -> None:
    find_column(raw, "orders", "order_id")["nullable"] = True


def unknown_hint(raw: dict[str, Any]) -> None:
    find_column(raw, "orders", "quantity")["gen"]["distrbution"] = "uniform"


def cycle_with_weights(raw: dict[str, Any]) -> None:
    find_column(raw, "events", "event_name")["gen"].update(distribution="cycle", weights=[1, 1, 1])


def uniform_without_range(raw: dict[str, Any]) -> None:
    find_column(raw, "orders", "quantity")["gen"] = {"distribution": "uniform"}


@pytest.mark.parametrize(
    ("breakage", "message"),
    [
        (null_rate_without_nullable, "null_rate needs nullable"),
        (string_without_generator, "exactly one of values, faker or pattern"),
        (struct_shape_missing_field, "shape does not match"),
        (array_shape_wrong_element, "shape does not match"),
        (edge_rows_too_many, "edge_rows"),
        (after_wrong_type, "same type"),
        (nullable_primary_key, "non-nullable sequences"),
        (unknown_hint, "Extra inputs are not permitted"),
        (uniform_without_range, "needs min and max"),
        (cycle_with_weights, "cycle needs values and no weights"),
    ],
)
def test_invalid_tables_are_rejected(breakage: Any, message: str) -> None:
    raw = fixture_dict()
    breakage(raw)

    with pytest.raises(ValidationError, match=message):
        for table_raw in raw["tables"]:
            TableSpec.model_validate(table_raw)


def test_fixture_tables_are_valid() -> None:
    for table_raw in fixture_dict()["tables"]:
        TableSpec.model_validate(table_raw)
