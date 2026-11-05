"""Edge-profile injectors: small, targeted changes that break SQL which is only
accidentally right on realistic data.

Each injector edits the generated tables in place and returns the tables where
its case is now guaranteed to exist. Injectors never touch primary keys, only
set a foreign key to another existing parent or to NULL where allowed, and
only duplicate rows in tables without a primary key, so integrity holds.
"""

import random
from collections.abc import Callable
from typing import Any

from sqlbench.datagen.tables import TableData
from sqlbench.datagen.values import derive_seed
from sqlbench.schema.spec import ColumnSpec, SchemaSpec, TableSpec
from sqlbench.schema.types import NUMERIC_KINDS, TypeKind

Tables = dict[str, TableData]
Injector = Callable[[SchemaSpec, Tables, random.Random], list[str]]

# The ranks tested by "top N" questions: a tie at rank 1-2 and at the rank 10-11 cut.
TIE_RANKS = (1, 10)
# Rows in the top 11 of a measure stay unique, so duplicates never shift those ranks.
PROTECTED_TOP_ROWS = 11
EMPTY_GROUP_SHARE = 0.05
NULL_SHARE = 0.05
DUPLICATE_SHARE = 0.01


def apply_edge_injectors(schema: SchemaSpec, tables: Tables, seed: int) -> None:
    rng = random.Random(derive_seed(seed, "edge", schema.name, "injectors"))
    for name, injector in INJECTORS:
        for table_name in injector(schema, tables, rng):
            tables[table_name].injectors.append(name)


def is_key(table: TableSpec, column: ColumnSpec) -> bool:
    return column.name == table.primary_key or column.references is not None


def rankable_columns(table: TableSpec) -> list[ColumnSpec]:
    """Numeric measures and timestamps, the columns top-N questions sort by."""
    rankable_kinds = NUMERIC_KINDS | {TypeKind.TIMESTAMP_NTZ, TypeKind.TIMESTAMP_TZ}
    columns: list[ColumnSpec] = []
    for column in table.columns:
        if column.logical_type.kind in rankable_kinds and not is_key(table, column):
            columns.append(column)
    return columns


def non_null_rows(values: list[Any]) -> list[int]:
    return [index for index, value in enumerate(values) if value is not None]


def rows_by_rank(values: list[Any]) -> list[int]:
    """Row indexes of non-null values, largest first. Ties keep row order."""
    rows = non_null_rows(values)
    rows.sort(key=lambda index: values[index], reverse=True)
    return rows


def repair_after_order(table: TableData) -> None:
    """Move after-columns forward when an injector moved their anchor past them."""
    for column in table.spec.columns:
        if column.gen.after is None:
            continue
        anchors = table.columns[column.gen.after]
        values = table.columns[column.name]
        for index, anchor in enumerate(anchors):
            if anchor is not None and values[index] is not None and values[index] < anchor:
                values[index] = anchor


def inject_empty_groups(schema: SchemaSpec, tables: Tables, rng: random.Random) -> list[str]:
    """Reserve a few parent rows (customers with no orders) that no child refers to."""
    changed: list[str] = []
    for parent in schema.tables:
        children: list[tuple[str, str]] = []
        for child in schema.tables:
            for column in child.columns:
                if column.references == f"{parent.name}.{parent.primary_key}":
                    children.append((child.name, column.name))
        if not children:
            continue
        assert parent.primary_key is not None
        keys = tables[parent.name].columns[parent.primary_key]
        reserved = set(rng.sample(keys, max(1, int(len(keys) * EMPTY_GROUP_SHARE))))
        allowed = [key for key in keys if key not in reserved]
        for child_name, column_name in children:
            values = tables[child_name].columns[column_name]
            for index, value in enumerate(values):
                if value in reserved:
                    values[index] = rng.choice(allowed)
        changed.append(parent.name)
    return changed


def inject_nulls(schema: SchemaSpec, tables: Tables, rng: random.Random) -> list[str]:
    """Every nullable column gets at least NULL_SHARE NULLs, including foreign keys."""
    changed: list[str] = []
    for table in schema.tables:
        data = tables[table.name]
        nullable = [column for column in table.columns if column.nullable]
        for column in nullable:
            values = data.columns[column.name]
            wanted = max(1, int(len(values) * NULL_SHARE)) - values.count(None)
            if wanted > 0:
                for index in rng.sample(non_null_rows(values), wanted):
                    values[index] = None
        if nullable:
            changed.append(table.name)
    return changed


def inject_ties(schema: SchemaSpec, tables: Tables, rng: random.Random) -> list[str]:
    """Copy the value at rank N onto rank N+1, for each rank in TIE_RANKS."""
    changed: list[str] = []
    for table in schema.tables:
        data = tables[table.name]
        touched = False
        for column in rankable_columns(table):
            values = data.columns[column.name]
            if len(non_null_rows(values)) <= max(TIE_RANKS):
                continue
            for rank in TIE_RANKS:
                ranked = rows_by_rank(values)
                values[ranked[rank]] = values[ranked[rank - 1]]
            touched = True
        if touched:
            # Raising a timestamp anchor can pass its after-column.
            repair_after_order(data)
            changed.append(table.name)
    return changed


def inject_duplicates(schema: SchemaSpec, tables: Tables, rng: random.Random) -> list[str]:
    """Append exact copies of some rows to tables without a primary key."""
    changed: list[str] = []
    for table in schema.tables:
        if table.primary_key is not None:
            continue
        data = tables[table.name]
        protected: set[int] = set()
        for column in rankable_columns(table):
            protected.update(rows_by_rank(data.columns[column.name])[:PROTECTED_TOP_ROWS])
        candidates = [index for index in range(data.row_count) if index not in protected]
        copies = rng.sample(candidates, max(1, int(data.row_count * DUPLICATE_SHARE)))
        for values in data.columns.values():
            for index in copies:
                values.append(values[index])
        changed.append(table.name)
    return changed


# Order matters: duplicates come last so copies match their originals exactly.
INJECTORS: list[tuple[str, Injector]] = [
    ("empty_groups", inject_empty_groups),
    ("nulls", inject_nulls),
    ("ties", inject_ties),
    ("duplicates", inject_duplicates),
]
