"""Loading through the shared driver, into DuckDB."""

import datetime
import json
from pathlib import Path

import duckdb
import pytest
from typer.testing import CliRunner

from sqlbench.cli import app
from sqlbench.datagen.generator import generate_dataset
from sqlbench.datagen.tables import Profile
from sqlbench.load.dataset import LoadError, load_dataset, session_context, target_schema
from sqlbench.load.duckdb_loader import DuckDBLoader, duckdb_type
from sqlbench.schema.types import parse_type
from sqlbench.warehouses.duckdb_warehouse import DuckDBWarehouse

FIXTURE = Path(__file__).parent / "fixtures" / "mini_shop.yaml"


@pytest.fixture
def generated(tmp_path: Path) -> Path:
    for profile in Profile:
        generate_dataset(FIXTURE, profile, 5, tmp_path / "data", tmp_path / "manifests")
    return tmp_path


def load(root: Path, profile: Profile) -> None:
    loader = DuckDBLoader(root / "sqlbench.duckdb")
    try:
        load_dataset(loader, FIXTURE, profile, root / "data", root / "manifests")
    finally:
        loader.close()


def query(root: Path, profile: Profile, sql: str) -> list[tuple[object, ...]]:
    connection = duckdb.connect(str(root / "sqlbench.duckdb"), read_only=True)
    warehouse = DuckDBWarehouse(connection, session_context("mini_shop", profile))
    try:
        result = warehouse.execute(sql, timeout_s=30)
    finally:
        warehouse.close()
    assert result.error is None, result.error
    return result.rows


def test_profiles_get_their_own_schema_with_the_same_table_names() -> None:
    assert target_schema("ecommerce", Profile.MAIN) == "ecommerce"
    assert target_schema("ecommerce", Profile.EDGE) == "ecommerce__edge"
    assert session_context("ecommerce", Profile.EDGE).database == "sqlbench"


@pytest.mark.parametrize(
    ("logical", "expected"),
    [
        ("decimal(10,2)", "DECIMAL(10,2)"),
        ("array<string>", "VARCHAR[]"),
        ("map<string,int>", "MAP(VARCHAR, INTEGER)"),
        ("struct<a:int,b:timestamp_tz>", "STRUCT(a INTEGER, b TIMESTAMPTZ)"),
        ("variant", "JSON"),
    ],
)
def test_duckdb_type_names(logical: str, expected: str) -> None:
    assert duckdb_type(parse_type(logical)) == expected


def test_same_unqualified_query_reads_main_or_edge(generated: Path) -> None:
    load(generated, Profile.MAIN)
    load(generated, Profile.EDGE)

    assert query(generated, Profile.MAIN, "SELECT count(*) FROM orders") == [(1000,)]
    assert query(generated, Profile.EDGE, "SELECT count(*) FROM orders") == [(400,)]


def test_loading_twice_gives_the_same_tables(generated: Path) -> None:
    load(generated, Profile.EDGE)
    first = query(generated, Profile.EDGE, "SELECT count(*), sum(total) FROM orders")
    load(generated, Profile.EDGE)

    assert query(generated, Profile.EDGE, "SELECT count(*), sum(total) FROM orders") == first


def test_json_null_stays_apart_from_sql_null(generated: Path) -> None:
    load(generated, Profile.EDGE)
    manifest = json.loads((generated / "manifests" / "edge" / "mini_shop.json").read_text())
    assert "semi_structured" in manifest["tables"]["orders"]["injectors"]

    rows = query(
        generated,
        Profile.EDGE,
        "SELECT count(*) FILTER (WHERE attributes IS NULL), "
        "count(*) FILTER (WHERE json_type(attributes) = 'NULL'), "
        "count(*) FILTER (WHERE attributes->>'$.channel' IS NOT NULL) FROM orders",
    )
    sql_nulls, json_nulls, with_channel = rows[0]
    assert isinstance(sql_nulls, int) and sql_nulls >= 1
    assert json_nulls == 1
    assert isinstance(with_channel, int) and with_channel > 300


def test_nested_and_temporal_types_survive(generated: Path) -> None:
    load(generated, Profile.MAIN)

    rows = query(
        generated,
        Profile.MAIN,
        "SELECT (SELECT any_value(typeof(specs)) FROM orders), "
        "(SELECT any_value(typeof(address)) FROM customers), "
        "(SELECT any_value(typeof(tags)) FROM customers), "
        "(SELECT min(event_ts) FROM events)",
    )
    specs_type, address_type, tags_type, first_event = rows[0]
    assert specs_type == "MAP(VARCHAR, VARCHAR)"
    assert address_type == "STRUCT(city VARCHAR, zip VARCHAR)"
    assert tags_type == "VARCHAR[]"
    assert isinstance(first_event, datetime.datetime)
    assert first_event.utcoffset() == datetime.timedelta(0)


def test_files_that_differ_from_the_manifest_are_refused(generated: Path) -> None:
    parquet = generated / "data" / "edge" / "mini_shop" / "orders.parquet"
    parquet.write_bytes(parquet.read_bytes()[:-10])
    loader = DuckDBLoader(generated / "sqlbench.duckdb")

    with pytest.raises(LoadError, match="orders.parquet is missing or differs"):
        load_dataset(loader, FIXTURE, Profile.EDGE, generated / "data", generated / "manifests")
    loader.close()


def test_cli_loads_duckdb_without_asking(generated: Path) -> None:
    result = CliRunner().invoke(
        app,
        [
            "data",
            "load",
            "--warehouse",
            "duckdb",
            "--schema",
            "mini_shop",
            "--dir",
            str(FIXTURE.parent),
            "--data-dir",
            str(generated / "data"),
            "--manifest-dir",
            str(generated / "manifests"),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "mini_shop (main): loaded into duckdb" in result.output
    assert "mini_shop (edge): loaded into duckdb" in result.output
    assert "Estimated cost" not in result.output


def test_cli_prints_a_cost_estimate_and_stops_when_declined(generated: Path) -> None:
    result = CliRunner().invoke(
        app,
        ["data", "load", "--warehouse", "snowflake", "--schema", "mini_shop"]
        + ["--dir", str(FIXTURE.parent), "--data-dir", str(generated / "data")],
        input="n\n",
    )

    assert result.exit_code == 1
    assert "Estimated cost: 6 tables" in result.output
    assert "credits" in result.output
    assert "loaded into" not in result.output
