"""Checks on the real schemas in schemas/. New schema files are picked up automatically."""

import json
from pathlib import Path

import pytest
import sqlglot

from sqlbench.datagen.generator import (
    GENERATOR_VERSION,
    file_sha256,
    find_integrity_problems,
    generate_dataset,
    generate_tables,
)
from sqlbench.datagen.tables import Profile
from sqlbench.schema.ddl import create_table_statements
from sqlbench.schema.spec import load_schema
from sqlbench.warehouses.base import WarehouseName

REPO = Path(__file__).parent.parent
SCHEMA_PATHS = sorted((REPO / "schemas").glob("*.yaml"))
SCHEMA_IDS = [path.stem for path in SCHEMA_PATHS]


@pytest.mark.parametrize("path", SCHEMA_PATHS, ids=SCHEMA_IDS)
def test_schema_shape_matches_the_plan(path: Path) -> None:
    schema = load_schema(path)

    assert 4 <= len(schema.tables) <= 7
    largest = max(table.rows for table in schema.tables)
    assert 400_000 <= largest <= 600_000


@pytest.mark.parametrize("path", SCHEMA_PATHS, ids=SCHEMA_IDS)
def test_edge_profile_keeps_integrity_and_applies_injectors(path: Path) -> None:
    schema = load_schema(path)
    tables = generate_tables(schema, Profile.EDGE, seed=1)

    assert find_integrity_problems(schema, tables) == []
    applied = {name for table in tables.values() for name in table.injectors}
    expected = {
        "empty_groups",
        "nulls",
        "boundary_dates",
        "semi_structured",
        "messy_strings",
        "ties",
    }
    if any(table.primary_key is None for table in schema.tables):
        expected.add("duplicates")
    assert expected <= applied


@pytest.mark.parametrize("path", SCHEMA_PATHS, ids=SCHEMA_IDS)
@pytest.mark.parametrize("profile", list(Profile))
def test_committed_manifest_matches_the_spec(path: Path, profile: Profile) -> None:
    manifest = json.loads((REPO / "manifests" / profile / f"{path.stem}.json").read_text())

    # A stale manifest means the YAML or generator changed without regenerating.
    assert manifest["spec_sha256"] == file_sha256(path)
    assert manifest["generator_version"] == GENERATOR_VERSION


@pytest.mark.parametrize("path", SCHEMA_PATHS, ids=SCHEMA_IDS)
def test_edge_files_reproduce_the_committed_hashes(path: Path, tmp_path: Path) -> None:
    committed = json.loads((REPO / "manifests" / "edge" / f"{path.stem}.json").read_text())

    regenerated = generate_dataset(
        path, Profile.EDGE, committed["seed"], tmp_path / "data", tmp_path / "manifests"
    )

    for table_name, table in regenerated.tables.items():
        assert table.sha256 == committed["tables"][table_name]["sha256"], table_name


@pytest.mark.parametrize("path", SCHEMA_PATHS, ids=SCHEMA_IDS)
@pytest.mark.parametrize("dialect", [WarehouseName.SNOWFLAKE, WarehouseName.DATABRICKS])
def test_ddl_parses(path: Path, dialect: WarehouseName) -> None:
    schema = load_schema(path)

    for statement in create_table_statements(schema, dialect, schema.name):
        sqlglot.parse_one(statement, read=dialect.value)
