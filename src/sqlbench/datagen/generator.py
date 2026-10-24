"""Generate every table of a schema and write Parquet files plus a manifest."""

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel

from sqlbench.datagen.edge import apply_edge_injectors
from sqlbench.datagen.tables import Profile, TableData
from sqlbench.datagen.values import (
    JsonNull,
    derive_seed,
    generate_after,
    generate_self_references,
    generate_series,
    generate_values,
)
from sqlbench.schema.spec import (
    ColumnSpec,
    SchemaSpec,
    TableSpec,
    is_self_reference,
    load_schema,
)
from sqlbench.schema.types import NUMERIC_KINDS, TEMPORAL_KINDS, LogicalType, TypeKind

# Bump when the same seed would produce different data, so old manifests are
# recognizably stale.
GENERATOR_VERSION = 1


def generate_tables(schema: SchemaSpec, profile: Profile, seed: int) -> dict[str, TableData]:
    tables: dict[str, TableData] = {}
    for table in schema.tables_in_dependency_order():
        row_count = table.rows if profile == Profile.MAIN else table.edge_rows
        values_by_column: dict[str, list[Any]] = {}
        # Columns that depend on other columns of the row wait for them.
        for column in table.columns:
            if column.gen.after is not None or is_self_reference(table, column):
                continue
            if table.series is not None and column.name in (table.series.time, table.series.end):
                continue
            column_seed = derive_seed(seed, profile, schema.name, table.name, column.name)
            parent_keys = None
            if column.references is not None:
                parent_table, parent_column = column.references.split(".")
                parent_keys = tables[parent_table].columns[parent_column]
            values_by_column[column.name] = generate_values(
                column, row_count, column_seed, parent_keys
            )
        if table.series is not None:
            series_seed = derive_seed(seed, profile, schema.name, table.name, table.series.time)
            keys = values_by_column[table.series.key]
            values_by_column.update(generate_series(table, keys, series_seed))
        for column in table.columns:
            if column.gen.after is None:
                continue
            column_seed = derive_seed(seed, profile, schema.name, table.name, column.name)
            anchors = values_by_column[column.gen.after]
            values_by_column[column.name] = generate_after(column, anchors, column_seed)
        for column in table.columns:
            if not is_self_reference(table, column):
                continue
            column_seed = derive_seed(seed, profile, schema.name, table.name, column.name)
            assert table.primary_key is not None
            keys = values_by_column[table.primary_key]
            values_by_column[column.name] = generate_self_references(column, keys, column_seed)

        ordered = {column.name: values_by_column[column.name] for column in table.columns}
        tables[table.name] = TableData(table, ordered)
    if profile == Profile.EDGE:
        apply_edge_injectors(schema, tables, seed)
    return tables


def arrow_type(logical_type: LogicalType) -> pa.DataType:
    kind = logical_type.kind
    if kind == TypeKind.DECIMAL:
        return pa.decimal128(logical_type.precision, logical_type.scale)
    if kind == TypeKind.ARRAY:
        assert logical_type.element is not None
        return pa.list_(arrow_type(logical_type.element))
    if kind == TypeKind.MAP:
        assert logical_type.key is not None and logical_type.element is not None
        return pa.map_(arrow_type(logical_type.key), arrow_type(logical_type.element))
    if kind == TypeKind.STRUCT:
        return pa.struct([(f.name, arrow_type(f.type)) for f in logical_type.fields])
    simple_types = {
        TypeKind.INT: pa.int32(),
        TypeKind.BIGINT: pa.int64(),
        TypeKind.DOUBLE: pa.float64(),
        TypeKind.STRING: pa.string(),
        TypeKind.BOOL: pa.bool_(),
        TypeKind.DATE: pa.date32(),
        TypeKind.TIMESTAMP_NTZ: pa.timestamp("us"),
        TypeKind.TIMESTAMP_TZ: pa.timestamp("us", tz="UTC"),
        # Parquet has no VARIANT, so JSON text is stored and parsed at load time.
        TypeKind.VARIANT: pa.string(),
    }
    return simple_types[kind]


def find_integrity_problems(schema: SchemaSpec, tables: dict[str, TableData]) -> list[str]:
    """Primary key, NOT NULL, foreign key and after-ordering violations, as messages."""
    problems: list[str] = []
    for table in schema.tables:
        columns = tables[table.name].columns
        for column in table.columns:
            values = columns[column.name]
            where = f"{table.name}.{column.name}"
            if not column.nullable and None in values:
                problems.append(f"{where}: NULL in a non-nullable column")
            if column.references is not None:
                parent_table, parent_column = column.references.split(".")
                parent_keys = set(tables[parent_table].columns[parent_column])
                orphans = {value for value in values if value is not None} - parent_keys
                if orphans:
                    problems.append(f"{where}: {len(orphans)} values missing from {parent_table}")
            if column.gen.after is not None:
                anchors = columns[column.gen.after]
                for anchor, value in zip(anchors, values, strict=True):
                    if anchor is not None and value is not None and value < anchor:
                        problems.append(f"{where}: earlier than {column.gen.after}")
                        break
            if is_self_reference(table, column):
                assert table.primary_key is not None
                own_keys = columns[table.primary_key]
                for own_key, parent in zip(own_keys, values, strict=True):
                    # Parents come from earlier rows; anything else could form a cycle.
                    if parent is not None and parent >= own_key:
                        problems.append(f"{where}: points at a later row")
                        break
        if table.primary_key is not None:
            keys = columns[table.primary_key]
            if len(set(keys)) != len(keys):
                problems.append(f"{table.name}.{table.primary_key}: duplicate primary keys")
    return problems


def to_arrow_values(logical_type: LogicalType, values: list[Any]) -> list[Any]:
    """Adapt Python values to what pyarrow expects for the column type."""
    kind = logical_type.kind
    if kind == TypeKind.VARIANT:
        converted: list[Any] = []
        for value in values:
            if value is None:
                converted.append(None)
            elif isinstance(value, JsonNull):
                converted.append("null")
            else:
                converted.append(json.dumps(value, separators=(",", ":")))
        return converted
    if kind == TypeKind.MAP:
        return [None if value is None else list(value.items()) for value in values]
    return values


def to_arrow_table(table: TableData) -> pa.Table:
    arrays = []
    fields = []
    for column in table.spec.columns:
        column_type = arrow_type(column.logical_type)
        values = to_arrow_values(column.logical_type, table.columns[column.name])
        arrays.append(pa.array(values, type=column_type))
        fields.append(pa.field(column.name, column_type, nullable=column.nullable))
    return pa.Table.from_arrays(arrays, schema=pa.schema(fields))


class ColumnStats(BaseModel):
    """Aggregates of one column, recomputed in the warehouse after loading."""

    nulls: int
    # Sum of a numeric column: exact text for integers and decimals, repr for doubles.
    total: str | None = None
    # Distinct non-null values, for primary and foreign key columns.
    distinct: int | None = None
    # Earliest and latest value of a date or timestamp column, ISO 8601.
    min: str | None = None
    max: str | None = None


def column_stats(table: TableSpec, column: ColumnSpec, values: list[Any]) -> ColumnStats:
    present = [value for value in values if value is not None]
    stats = ColumnStats(nulls=len(values) - len(present))
    kind = column.logical_type.kind
    if kind == TypeKind.DOUBLE:
        stats.total = repr(math.fsum(present))
    elif kind in NUMERIC_KINDS:
        stats.total = str(sum(present))
    if column.name == table.primary_key or column.references is not None:
        stats.distinct = len(set(present))
    if kind in TEMPORAL_KINDS and present:
        stats.min = min(present).isoformat()
        stats.max = max(present).isoformat()
    return stats


class TableManifest(BaseModel):
    rows: int
    sha256: str
    injectors: list[str]
    columns: dict[str, ColumnStats]


class Manifest(BaseModel):
    """Everything needed to tell whether two datasets are the same."""

    schema_name: str
    profile: Profile
    seed: int
    generator_version: int
    spec_sha256: str
    # Parquet bytes depend on the writer version, so file hashes only match across
    # machines with the same pyarrow.
    pyarrow_version: str
    tables: dict[str, TableManifest]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def generate_dataset(
    schema_path: Path, profile: Profile, seed: int, data_dir: Path, manifest_dir: Path
) -> Manifest:
    """Write data/<profile>/<schema>/<table>.parquet and manifests/<profile>/<schema>.json."""
    schema = load_schema(schema_path)
    tables = generate_tables(schema, profile, seed)
    problems = find_integrity_problems(schema, tables)
    if problems:
        raise ValueError(f"generated {schema.name} ({profile}) is inconsistent: {problems}")

    output_dir = data_dir / profile / schema.name
    output_dir.mkdir(parents=True, exist_ok=True)
    table_manifests: dict[str, TableManifest] = {}
    for table in schema.tables:
        data = tables[table.name]
        path = output_dir / f"{table.name}.parquet"
        pq.write_table(to_arrow_table(data), path, compression="zstd")
        stats = {}
        for column in table.columns:
            stats[column.name] = column_stats(table, column, data.columns[column.name])
        table_manifests[table.name] = TableManifest(
            rows=data.row_count,
            sha256=file_sha256(path),
            injectors=data.injectors,
            columns=stats,
        )

    manifest = Manifest(
        schema_name=schema.name,
        profile=profile,
        seed=seed,
        generator_version=GENERATOR_VERSION,
        spec_sha256=file_sha256(schema_path),
        pyarrow_version=pa.__version__,
        tables=table_manifests,
    )
    manifest_path = manifest_dir / profile / f"{schema.name}.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(manifest.model_dump_json(indent=2) + "\n")
    return manifest
