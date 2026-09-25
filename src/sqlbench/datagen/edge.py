"""Edge-profile injectors: small, targeted changes that break SQL which is only
accidentally right on realistic data.

Each injector edits the generated tables in place and returns the tables where
its case is now guaranteed to exist. Injectors never touch primary keys, only
set a foreign key to another existing parent or to NULL where allowed, and
only duplicate rows in tables without a primary key, so integrity holds.
"""

import datetime
import random
from collections.abc import Callable
from typing import Any

from sqlbench.datagen.tables import TableData
from sqlbench.datagen.values import JSON_NULL, derive_seed
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

UTC = datetime.UTC
BOUNDARY_DATES = [
    datetime.date(2023, 12, 31),
    datetime.date(2024, 1, 1),
    datetime.date(2024, 1, 31),
    datetime.date(2024, 2, 29),
    datetime.date(2024, 3, 1),
    datetime.date(2024, 4, 30),
    datetime.date(2024, 12, 31),
]
BOUNDARY_TIMESTAMPS = [
    datetime.datetime(2023, 12, 31, 23, 59, 59, 999999),
    datetime.datetime(2024, 1, 1, 0, 0, 0),
    datetime.datetime(2024, 1, 31, 23, 59, 59, 999999),
    datetime.datetime(2024, 2, 29, 0, 0, 0),
    datetime.datetime(2024, 2, 29, 23, 59, 59),
    datetime.datetime(2024, 3, 1, 0, 0, 0),
]
# US DST changes in 2024, as UTC instants on either side of each jump:
# 01:59:59 EST -> 03:00 EDT on Mar 10, and 01:59:59 EDT -> 01:00 EST on Nov 3.
DST_INSTANTS = [
    datetime.datetime(2024, 3, 10, 6, 59, 59, tzinfo=UTC),
    datetime.datetime(2024, 3, 10, 7, 0, 0, tzinfo=UTC),
    datetime.datetime(2024, 11, 3, 5, 59, 59, tzinfo=UTC),
    datetime.datetime(2024, 11, 3, 6, 0, 0, tzinfo=UTC),
]


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


def inject_boundary_dates(schema: SchemaSpec, tables: Tables, rng: random.Random) -> list[str]:
    """Month ends, Feb 29, the year change, midnight, and DST jumps for timestamp_tz."""
    changed: list[str] = []
    for table in schema.tables:
        data = tables[table.name]
        touched = False
        for column in table.columns:
            # Columns placed after an anchor follow it; repair_after_order keeps them valid.
            if column.gen.after is not None:
                continue
            boundaries = boundary_values(column.logical_type.kind)
            values = data.columns[column.name]
            rows = non_null_rows(values)
            if not boundaries or len(rows) < len(boundaries):
                continue
            for index, boundary in zip(rng.sample(rows, len(boundaries)), boundaries, strict=True):
                values[index] = boundary
            touched = True
        if touched:
            repair_after_order(data)
            changed.append(table.name)
    return changed


def boundary_values(kind: TypeKind) -> list[Any]:
    if kind == TypeKind.DATE:
        return list(BOUNDARY_DATES)
    if kind == TypeKind.TIMESTAMP_NTZ:
        return list(BOUNDARY_TIMESTAMPS)
    if kind == TypeKind.TIMESTAMP_TZ:
        in_utc = [timestamp.replace(tzinfo=UTC) for timestamp in BOUNDARY_TIMESTAMPS]
        return in_utc + DST_INSTANTS
    return []


def inject_semi_structured(schema: SchemaSpec, tables: Tables, rng: random.Random) -> list[str]:
    """Missing keys, JSON null vs SQL NULL, empty and single-element arrays, empty maps."""
    changed: list[str] = []
    for table in schema.tables:
        touched = False
        for column in table.columns:
            values = tables[table.name].columns[column.name]
            rows = non_null_rows(values)
            rng.shuffle(rows)
            if edit_nested_column(column, values, rows):
                touched = True
        if touched:
            changed.append(table.name)
    return changed


def edit_nested_column(column: ColumnSpec, values: list[Any], free_rows: list[int]) -> bool:
    """Apply each nested edge case to its own row, taken from free_rows."""
    edits = nested_edits(column)
    if not edits or len(free_rows) < len(edits):
        return False
    for edit, key in edits:
        index = take_row(free_rows, values, edit, key)
        if index is None:
            return False
        values[index] = apply_nested_edit(values[index], edit, key)
    return True


def nested_edits(column: ColumnSpec) -> list[tuple[str, str | None]]:
    kind = column.logical_type.kind
    shape = column.gen.shape
    edits: list[tuple[str, str | None]] = []
    if kind == TypeKind.VARIANT and shape is not None and shape.type == "object":
        for name, field_shape in shape.fields.items():
            edits.append(("drop_key", name))
            edits.append(("json_null_key", name))
            if field_shape.type == "array":
                edits.append(("empty_array", name))
                edits.append(("single_item_array", name))
        edits.append(("json_null", None))
    elif kind == TypeKind.ARRAY:
        edits = [("empty_array", None), ("single_item_array", None)]
    elif kind == TypeKind.MAP:
        edits = [("empty_map", None)]
    elif kind == TypeKind.STRUCT:
        edits = [("null_field", column.logical_type.fields[0].name)]
    return edits


def take_row(free_rows: list[int], values: list[Any], edit: str, key: str | None) -> int | None:
    """Pop a row the edit can apply to; a single-item array needs a non-empty one."""
    for position, index in enumerate(free_rows):
        if edit == "single_item_array" and not array_at(values[index], key):
            continue
        return free_rows.pop(position)
    return None


def array_at(value: Any, key: str | None) -> list[Any]:
    if key is None:
        return list(value)
    found = value.get(key) if isinstance(value, dict) else None
    return found if isinstance(found, list) else []


def apply_nested_edit(value: Any, edit: str, key: str | None) -> Any:
    if edit == "json_null":
        return JSON_NULL
    if edit == "empty_map":
        return {}
    if edit == "empty_array" and key is None:
        return []
    if edit == "single_item_array" and key is None:
        return value[:1]
    assert key is not None
    edited = dict(value)
    if edit == "drop_key":
        edited.pop(key, None)
    elif edit == "single_item_array":
        edited[key] = edited[key][:1]
    elif edit == "empty_array":
        edited[key] = []
    else:
        # json_null_key on a VARIANT, or null_field on a struct.
        edited[key] = None
    return edited


def inject_messy_strings(schema: SchemaSpec, tables: Tables, rng: random.Random) -> list[str]:
    """Leading and trailing spaces, upper case and mixed case on a few rows per column."""
    changed: list[str] = []
    variants: list[Callable[[str], str]] = [
        lambda text: " " + text,
        lambda text: text + " ",
        lambda text: text.upper(),
        mixed_case,
    ]
    for table in schema.tables:
        touched = False
        for column in table.columns:
            if column.logical_type.kind != TypeKind.STRING:
                continue
            values = tables[table.name].columns[column.name]
            rows = non_null_rows(values)
            if len(rows) < len(variants):
                continue
            for index, variant in zip(rng.sample(rows, len(variants)), variants, strict=True):
                values[index] = variant(values[index])
            touched = True
        if touched:
            changed.append(table.name)
    return changed


def mixed_case(text: str) -> str:
    characters: list[str] = []
    for position, character in enumerate(text):
        characters.append(character.upper() if position % 2 else character.lower())
    return "".join(characters)


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


# Order matters: ties come after every other value change so nothing breaks
# them, and duplicates come last so copies match their originals exactly.
INJECTORS: list[tuple[str, Injector]] = [
    ("empty_groups", inject_empty_groups),
    ("nulls", inject_nulls),
    ("boundary_dates", inject_boundary_dates),
    ("semi_structured", inject_semi_structured),
    ("messy_strings", inject_messy_strings),
    ("ties", inject_ties),
    ("duplicates", inject_duplicates),
]
