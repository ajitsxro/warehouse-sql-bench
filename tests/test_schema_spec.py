import copy
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from sqlbench.schema.spec import SchemaSpec, TableSpec, load_schema

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


def test_fixture_loads_with_parents_before_children() -> None:
    schema = load_schema(FIXTURE)

    ordered = [table.name for table in schema.tables_in_dependency_order()]
    assert ordered.index("customers") < ordered.index("orders")
    assert ordered.index("customers") < ordered.index("events")
    assert schema.table("events").primary_key is None
    assert str(schema.table("orders").column("specs").logical_type) == "map<string,string>"


def test_schema_name_must_match_file_name(tmp_path: Path) -> None:
    path = tmp_path / "other_name.yaml"
    path.write_text(FIXTURE.read_text())

    with pytest.raises(ValueError, match="must match the file name"):
        load_schema(path)


def break_fk_target(raw: dict[str, Any]) -> None:
    find_column(raw, "orders", "customer_id")["references"] = "customers.full_name"


def break_fk_table(raw: dict[str, Any]) -> None:
    find_column(raw, "orders", "customer_id")["references"] = "accounts.account_id"


def break_fk_type(raw: dict[str, Any]) -> None:
    find_column(raw, "orders", "customer_id")["type"] = "int"


def make_cycle(raw: dict[str, Any]) -> None:
    customers = raw["tables"][0]
    customers["columns"].append(
        {"name": "last_order_id", "type": "bigint", "references": "orders.order_id"}
    )


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


@pytest.mark.parametrize(
    ("breakage", "message"),
    [
        (break_fk_target, "not the primary key"),
        (break_fk_table, "unknown table accounts"),
        (break_fk_type, "type differs"),
        (make_cycle, "cycle"),
    ],
)
def test_invalid_schemas_are_rejected(breakage: Any, message: str) -> None:
    raw = fixture_dict()
    breakage(raw)

    with pytest.raises(ValidationError, match=message):
        SchemaSpec.model_validate(raw)
