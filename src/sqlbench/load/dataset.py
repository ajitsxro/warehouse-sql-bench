"""Warehouse-independent loading: naming, file checks, and table order.

Each warehouse gets one database (a catalog on Databricks) named sqlbench,
with two schemas per benchmark schema: <name> for main and <name>__edge for
edge. Table names are the same in both, so one unqualified query runs against
either profile by switching the session schema.
"""

import json
from pathlib import Path
from typing import Protocol

from sqlbench.datagen.generator import Manifest, file_sha256
from sqlbench.datagen.tables import Profile
from sqlbench.schema.spec import SchemaSpec, TableSpec, load_schema
from sqlbench.warehouses.base import SessionContext, WarehouseName

DATABASE = "sqlbench"
EDGE_SUFFIX = "__edge"


class LoadError(Exception):
    """The local files don't match the committed manifest, so nothing was loaded."""


def target_schema(schema_name: str, profile: Profile) -> str:
    if profile == Profile.EDGE:
        return schema_name + EDGE_SUFFIX
    return schema_name


def session_context(schema_name: str, profile: Profile) -> SessionContext:
    return SessionContext(DATABASE, target_schema(schema_name, profile))


class Loader(Protocol):
    name: WarehouseName

    def reset_schema(self, target: str) -> None:
        """Drop the target schema with everything in it, and create it empty."""
        ...

    def load_table(self, target: str, table: TableSpec, parquet_path: Path) -> None:
        """Create the table and fill it from one Parquet file."""
        ...

    def grant_read(self, target: str) -> None:
        """Let the read-only principal SELECT from every table in the schema."""
        ...

    def close(self) -> None: ...


def read_manifest(manifest_dir: Path, schema_name: str, profile: Profile) -> Manifest:
    path = manifest_dir / profile / f"{schema_name}.json"
    return Manifest.model_validate(json.loads(path.read_text()))


def check_local_files(schema_path: Path, manifest: Manifest, dataset_dir: Path) -> dict[str, Path]:
    """Parquet path per table, after checking every file against the manifest."""
    if file_sha256(schema_path) != manifest.spec_sha256:
        raise LoadError(f"{schema_path.name} changed since its manifest was written")
    paths: dict[str, Path] = {}
    for table_name, table_manifest in manifest.tables.items():
        path = dataset_dir / f"{table_name}.parquet"
        if not path.exists() or file_sha256(path) != table_manifest.sha256:
            raise LoadError(
                f"{path} is missing or differs from the manifest; "
                f"run bench data generate --schema {manifest.schema_name} "
                f"--profile {manifest.profile}"
            )
        paths[table_name] = path
    return paths


def load_dataset(
    loader: Loader, schema_path: Path, profile: Profile, data_dir: Path, manifest_dir: Path
) -> SchemaSpec:
    """Replace one schema/profile in the warehouse with the committed data.

    Running it again gives the same result, because the target schema is
    dropped and rebuilt from files that must match the manifest's hashes.
    """
    schema = load_schema(schema_path)
    manifest = read_manifest(manifest_dir, schema.name, profile)
    paths = check_local_files(schema_path, manifest, data_dir / profile / schema.name)

    target = target_schema(schema.name, profile)
    loader.reset_schema(target)
    for table in schema.tables_in_dependency_order():
        loader.load_table(target, table, paths[table.name])
    loader.grant_read(target)
    return schema
