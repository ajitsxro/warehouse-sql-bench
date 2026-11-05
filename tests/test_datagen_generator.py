import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest
from typer.testing import CliRunner

from sqlbench.cli import app
from sqlbench.datagen.generator import (
    Manifest,
    arrow_type,
    find_integrity_problems,
    generate_dataset,
    generate_tables,
)
from sqlbench.datagen.tables import Profile
from sqlbench.schema.spec import load_schema

FIXTURE = Path(__file__).parent / "fixtures" / "mini_shop.yaml"
SCHEMA = load_schema(FIXTURE)


def generate(tmp_path: Path, profile: Profile, seed: int) -> Manifest:
    return generate_dataset(FIXTURE, profile, seed, tmp_path / "data", tmp_path / "manifests")


@pytest.mark.parametrize("profile", list(Profile))
def test_same_seed_writes_identical_files(tmp_path: Path, profile: Profile) -> None:
    first = generate(tmp_path / "first", profile, seed=5)
    second = generate(tmp_path / "second", profile, seed=5)
    other_seed = generate(tmp_path / "third", profile, seed=6)

    assert first == second
    for table_name in first.tables:
        assert first.tables[table_name].sha256 != other_seed.tables[table_name].sha256


@pytest.mark.parametrize("profile", list(Profile))
def test_generated_data_has_no_integrity_problems(profile: Profile) -> None:
    tables = generate_tables(SCHEMA, profile, seed=5)

    assert find_integrity_problems(SCHEMA, tables) == []


def test_integrity_check_catches_each_kind_of_violation() -> None:
    tables = generate_tables(SCHEMA, Profile.MAIN, seed=5)
    tables["customers"].columns["customer_id"][1] = tables["customers"].columns["customer_id"][0]
    tables["orders"].columns["customer_id"][0] = 999_999
    tables["orders"].columns["total"][0] = None
    shipped = tables["orders"].columns["shipped_at"]
    placed = tables["orders"].columns["placed_at"]
    index = shipped.index(next(value for value in shipped if value is not None))
    shipped[index] = placed[index].replace(year=placed[index].year - 1)

    problems = find_integrity_problems(SCHEMA, tables)

    assert any("duplicate primary keys" in problem for problem in problems)
    assert any("orders.customer_id: 1 values missing" in problem for problem in problems)
    assert any("orders.total: NULL" in problem for problem in problems)
    assert any("orders.shipped_at: earlier than placed_at" in problem for problem in problems)


def test_row_counts_follow_the_profile(tmp_path: Path) -> None:
    main = generate(tmp_path / "main", Profile.MAIN, seed=5)
    edge = generate(tmp_path / "edge", Profile.EDGE, seed=5)

    assert main.tables["orders"].rows == 1000
    assert edge.tables["orders"].rows == 400


def test_parquet_types_and_nested_values(tmp_path: Path) -> None:
    generate(tmp_path, Profile.MAIN, seed=5)
    orders = pq.read_table(tmp_path / "data" / "main" / "mini_shop" / "orders.parquet")

    for column in SCHEMA.table("orders").columns:
        field = orders.schema.field(column.name)
        assert field.type == arrow_type(column.logical_type)
        assert field.nullable == column.nullable
    attributes = [value for value in orders.column("attributes").to_pylist() if value]
    assert all("items" in json.loads(value) for value in attributes)
    specs = orders.column("specs").to_pylist()
    assert all(key in ("color", "size") for row in specs for key, _ in row)


def test_manifest_records_inputs(tmp_path: Path) -> None:
    manifest = generate(tmp_path, Profile.MAIN, seed=5)
    on_disk = json.loads((tmp_path / "manifests" / "main" / "mini_shop.json").read_text())

    assert on_disk["seed"] == 5
    assert on_disk["schema_name"] == "mini_shop"
    assert len(on_disk["spec_sha256"]) == 64
    assert set(on_disk["tables"]) == {"customers", "orders", "events"}
    assert Manifest.model_validate(on_disk) == manifest


def test_cli_needs_exactly_one_of_schema_or_all() -> None:
    neither = CliRunner().invoke(app, ["data", "generate", "--profile", "main"])
    both = CliRunner().invoke(
        app, ["data", "generate", "--profile", "main", "--schema", "x", "--all"]
    )

    assert neither.exit_code == 2
    assert both.exit_code == 2
