"""bench data check: manifest stats, value comparison, and a DuckDB round trip."""

import datetime
from decimal import Decimal
from pathlib import Path

import duckdb
import pytest
from typer.testing import CliRunner

from sqlbench.cli import app
from sqlbench.datagen.generator import generate_dataset
from sqlbench.datagen.tables import Profile
from sqlbench.load.check import check_dataset, same_value
from sqlbench.load.dataset import load_dataset, read_manifest, session_context
from sqlbench.load.duckdb_loader import DuckDBLoader
from sqlbench.schema.spec import load_schema
from sqlbench.schema.types import TypeKind
from sqlbench.warehouses.duckdb_warehouse import DuckDBWarehouse

FIXTURE = Path(__file__).parent / "fixtures" / "mini_shop.yaml"
SCHEMA = load_schema(FIXTURE)


@pytest.fixture
def loaded(tmp_path: Path) -> Path:
    loader = DuckDBLoader(tmp_path / "data" / "sqlbench.duckdb")
    for profile in Profile:
        generate_dataset(FIXTURE, profile, 5, tmp_path / "data", tmp_path / "manifests")
        load_dataset(loader, FIXTURE, profile, tmp_path / "data", tmp_path / "manifests")
    loader.close()
    return tmp_path


def check(root: Path, profile: Profile) -> list[str]:
    connection = duckdb.connect(str(root / "data" / "sqlbench.duckdb"), read_only=True)
    warehouse = DuckDBWarehouse(connection, session_context("mini_shop", profile))
    manifest = read_manifest(root / "manifests", "mini_shop", profile)
    try:
        return check_dataset(warehouse, SCHEMA, manifest, timeout_s=30)
    finally:
        warehouse.close()


@pytest.mark.parametrize(
    ("kind", "expected", "actual", "same"),
    [
        (TypeKind.DECIMAL, "3.75", Decimal("3.750"), True),
        (None, 12, Decimal("12"), True),
        (TypeKind.DOUBLE, "0.30000000000000004", 0.3, True),
        (TypeKind.DOUBLE, "0.3", 0.31, False),
        (
            TypeKind.TIMESTAMP_TZ,
            "2024-03-10T07:00:00+00:00",
            datetime.datetime(2024, 3, 10, 7, 0),
            True,
        ),
        (
            TypeKind.TIMESTAMP_TZ,
            "2024-03-10T07:00:00+00:00",
            datetime.datetime(
                2024, 3, 10, 0, 0, tzinfo=datetime.timezone(-datetime.timedelta(hours=7))
            ),
            True,
        ),
        (
            TypeKind.TIMESTAMP_TZ,
            "2024-03-10T07:00:00+00:00",
            datetime.datetime(2024, 3, 10, 0),
            False,
        ),
        (TypeKind.DATE, "2024-02-29", datetime.date(2024, 2, 29), True),
        (None, 5, None, False),
    ],
)
def test_same_value(kind: TypeKind | None, expected: object, actual: object, same: bool) -> None:
    assert same_value(kind, expected, actual) is same


@pytest.mark.parametrize("profile", list(Profile))
def test_a_fresh_load_matches_the_manifest(loaded: Path, profile: Profile) -> None:
    assert check(loaded, profile) == []


def test_missing_rows_and_changed_values_are_reported(loaded: Path) -> None:
    connection = duckdb.connect(str(loaded / "data" / "sqlbench.duckdb"))
    connection.execute('DELETE FROM "mini_shop__edge".events WHERE rowid < 3')
    connection.execute('UPDATE "mini_shop__edge".orders SET total = total + 1 WHERE order_id = 1')
    connection.execute('UPDATE "mini_shop__edge".orders SET attributes = NULL')
    connection.close()

    problems = check(loaded, Profile.EDGE)

    assert any(problem.startswith("events.rows: expected") for problem in problems)
    assert any(problem.startswith("orders.total.total: expected") for problem in problems)
    assert any(problem.startswith("orders.attributes.nulls: expected") for problem in problems)
    assert check(loaded, Profile.MAIN) == []


def test_cli_check_reports_matches(loaded: Path) -> None:
    result = CliRunner().invoke(
        app,
        ["data", "check", "--warehouse", "duckdb", "--schema", "mini_shop"]
        + ["--dir", str(FIXTURE.parent), "--data-dir", str(loaded / "data")]
        + ["--manifest-dir", str(loaded / "manifests")],
    )

    assert result.exit_code == 0, result.output
    assert "mini_shop (main): matches the manifest" in result.output
    assert "mini_shop (edge): matches the manifest" in result.output
