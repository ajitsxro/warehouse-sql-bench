"""Pydantic models for schemas/<name>.yaml, plus cross-table checks."""

import datetime
from functools import cached_property
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from sqlbench.schema.types import (
    INTEGER_KINDS,
    NUMERIC_KINDS,
    TEMPORAL_KINDS,
    LogicalType,
    TypeKind,
    TypeParseError,
    parse_type,
)

NAME_PATTERN = r"^[a-z][a-z0-9_]*$"
EDGE_MIN_ROWS = 100
EDGE_MAX_ROWS = 2000

# cycle assigns values in order, for small dimension tables with fixed rows.
Distribution = Literal["sequence", "uniform", "normal", "lognormal", "zipf", "cycle"]
Bound = float | datetime.datetime | datetime.date


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class JsonShape(StrictModel):
    """Shape of generated nested values: VARIANT JSON, arrays, structs and maps."""

    type: Literal["object", "array", "int", "number", "string", "bool"]
    fields: dict[str, "JsonShape"] = {}
    # Chance this key is left out of its parent object.
    optional: float = Field(default=0.0, ge=0, le=1)
    # Chance the value is JSON null rather than a value.
    null_rate: float = Field(default=0.0, ge=0, le=1)
    element: "JsonShape | None" = None
    min_items: int = Field(default=0, ge=0)
    max_items: int = Field(default=3, ge=0)
    min: float = 0
    max: float = 100
    scale: int = Field(default=2, ge=0)
    values: list[str] | None = None
    pattern: str | None = None

    @model_validator(mode="after")
    def check_shape(self) -> "JsonShape":
        if self.type == "object" and not self.fields:
            raise ValueError("object shapes need fields")
        if self.type == "array" and self.element is None:
            raise ValueError("array shapes need an element")
        if self.type == "string" and (self.values is None) == (self.pattern is None):
            raise ValueError("string shapes need exactly one of values or pattern")
        if self.min > self.max or self.min_items > self.max_items:
            raise ValueError("min must not exceed max")
        return self


class GenSpec(StrictModel):
    """Generator hints for one column. Which fields apply depends on the column type."""

    distribution: Distribution = "uniform"
    min: Bound | None = None
    max: Bound | None = None
    mean: float | None = None
    sigma: float | None = Field(default=None, gt=0)
    # Skew for zipf: higher means a few values dominate.
    exponent: float = Field(default=1.2, gt=1)
    values: list[str | int | float | bool] | None = None
    weights: list[float] | None = None
    faker: str | None = None
    pattern: str | None = None
    # How many distinct faker values to draw from.
    cardinality: int | None = Field(default=None, ge=1)
    null_rate: float = Field(default=0.0, ge=0, le=1)
    # Temporal columns: generate a value later than this column in the same row.
    after: str | None = None
    max_gap_days: float | None = Field(default=None, gt=0)
    shape: JsonShape | None = None

    @model_validator(mode="after")
    def check_weights(self) -> "GenSpec":
        if self.distribution == "cycle" and (self.values is None or self.weights is not None):
            raise ValueError("cycle needs values and no weights")
        if self.weights is not None:
            if self.values is None or len(self.weights) != len(self.values):
                raise ValueError("weights need values of the same length")
            if min(self.weights) < 0 or sum(self.weights) <= 0:
                raise ValueError("weights must be non-negative with a positive sum")
        return self


def shape_fits_type(shape: JsonShape, logical_type: LogicalType) -> bool:
    """Whether generated values of shape can be stored in a column of logical_type."""
    kind = logical_type.kind
    if kind == TypeKind.VARIANT:
        return True
    if kind == TypeKind.ARRAY:
        assert logical_type.element is not None
        return shape.element is not None and shape_fits_type(shape.element, logical_type.element)
    if kind == TypeKind.STRUCT:
        field_names = {field.name for field in logical_type.fields}
        if shape.type != "object" or set(shape.fields) != field_names:
            return False
        for field in logical_type.fields:
            if not shape_fits_type(shape.fields[field.name], field.type):
                return False
        return True
    if kind == TypeKind.MAP:
        assert logical_type.element is not None
        if shape.type != "object":
            return False
        for value_shape in shape.fields.values():
            if not shape_fits_type(value_shape, logical_type.element):
                return False
        return True
    scalar_shapes = {
        TypeKind.STRING: "string",
        TypeKind.BOOL: "bool",
        TypeKind.INT: "int",
        TypeKind.BIGINT: "int",
        TypeKind.DOUBLE: "number",
        TypeKind.DECIMAL: "number",
    }
    return scalar_shapes.get(kind) == shape.type


class ColumnSpec(StrictModel):
    name: str = Field(pattern=NAME_PATTERN)
    type: str
    nullable: bool = False
    description: str | None = None
    # Foreign key as "table.column"; the target must be that table's primary key.
    references: str | None = None
    gen: GenSpec = GenSpec()

    @field_validator("type")
    @classmethod
    def check_type(cls, text: str) -> str:
        try:
            parse_type(text)
        except TypeParseError as error:
            raise ValueError(str(error)) from None
        return text

    @cached_property
    def logical_type(self) -> LogicalType:
        return parse_type(self.type)

    @model_validator(mode="after")
    def check_gen_fits_type(self) -> "ColumnSpec":
        gen = self.gen
        kind = self.logical_type.kind
        if gen.null_rate > 0 and not self.nullable:
            raise ValueError("null_rate needs nullable: true")
        if self.references is not None:
            if kind not in INTEGER_KINDS:
                raise ValueError("foreign keys must be int or bigint")
            if gen.distribution not in ("uniform", "zipf"):
                raise ValueError("foreign keys pick parents with uniform or zipf")
            return self
        if kind in NUMERIC_KINDS:
            check_bounds(gen, (int, float), "numbers")
            check_numeric_gen(gen, kind)
        elif kind == TypeKind.STRING:
            choices = [gen.values, gen.faker, gen.pattern]
            if sum(choice is not None for choice in choices) != 1:
                raise ValueError("string columns need exactly one of values, faker or pattern")
        elif kind in TEMPORAL_KINDS:
            check_bounds(gen, (datetime.date,), "dates or timestamps")
            if gen.after is None and (gen.min is None or gen.max is None):
                raise ValueError("temporal columns need min and max, or after")
            if gen.after is not None and gen.max_gap_days is None:
                raise ValueError("after needs max_gap_days")
        elif kind != TypeKind.BOOL:
            if gen.shape is None:
                raise ValueError(f"{kind} columns need a shape")
            if not shape_fits_type(gen.shape, self.logical_type):
                raise ValueError(f"shape does not match type {self.logical_type}")
        return self


def check_bounds(gen: GenSpec, allowed: tuple[type, ...], description: str) -> None:
    for bound in (gen.min, gen.max):
        if bound is not None and not isinstance(bound, allowed):
            raise ValueError(f"min and max must be {description}")


def check_numeric_gen(gen: GenSpec, kind: TypeKind) -> None:
    if gen.values is not None:
        return
    if gen.distribution == "sequence" and kind not in INTEGER_KINDS:
        raise ValueError("sequence needs an int or bigint column")
    if gen.distribution in ("uniform", "zipf") and (gen.min is None or gen.max is None):
        raise ValueError(f"{gen.distribution} needs min and max")
    if gen.distribution in ("normal", "lognormal") and (gen.mean is None or gen.sigma is None):
        raise ValueError(f"{gen.distribution} needs mean and sigma")


class TableSpec(StrictModel):
    name: str = Field(pattern=NAME_PATTERN)
    description: str | None = None
    rows: int = Field(ge=1)
    edge_rows: int = Field(ge=1)
    # Event and fact tables leave this out; the edge profile duplicates their rows.
    primary_key: str | None = None
    columns: list[ColumnSpec] = Field(min_length=1)

    def column(self, name: str) -> ColumnSpec:
        for column in self.columns:
            if column.name == name:
                return column
        raise KeyError(f"{self.name} has no column {name}")

    @model_validator(mode="after")
    def check_columns(self) -> "TableSpec":
        names = [column.name for column in self.columns]
        if len(names) != len(set(names)):
            raise ValueError("duplicate column names")
        if self.rows >= EDGE_MIN_ROWS:
            if not EDGE_MIN_ROWS <= self.edge_rows <= min(EDGE_MAX_ROWS, self.rows):
                raise ValueError(f"edge_rows must be {EDGE_MIN_ROWS} to {EDGE_MAX_ROWS}")
        elif self.edge_rows != self.rows:
            raise ValueError(f"tables under {EDGE_MIN_ROWS} rows keep the same edge_rows")

        if self.primary_key is not None:
            if self.primary_key not in names:
                raise ValueError(f"primary key {self.primary_key} is not a column")
            key = self.column(self.primary_key)
            if key.nullable or key.gen.distribution != "sequence":
                raise ValueError("primary keys must be non-nullable sequences")

        for column in self.columns:
            anchor_name = column.gen.after
            if anchor_name is None:
                continue
            if anchor_name not in names or anchor_name == column.name:
                raise ValueError(f"{column.name}: after must name another column")
            if self.column(anchor_name).logical_type.kind != column.logical_type.kind:
                raise ValueError(f"{column.name}: after must name a column of the same type")
        return self


class SchemaSpec(StrictModel):
    name: str = Field(pattern=NAME_PATTERN)
    description: str
    tables: list[TableSpec] = Field(min_length=1)

    def table(self, name: str) -> TableSpec:
        for table in self.tables:
            if table.name == name:
                return table
        raise KeyError(f"{self.name} has no table {name}")

    @model_validator(mode="after")
    def check_references(self) -> "SchemaSpec":
        names = [table.name for table in self.tables]
        if len(names) != len(set(names)):
            raise ValueError("duplicate table names")
        for table in self.tables:
            for column in table.columns:
                if column.references is not None:
                    check_reference(self, table, column, column.references)
        self.tables_in_dependency_order()
        return self

    def tables_in_dependency_order(self) -> list[TableSpec]:
        """Parents before children, so foreign keys can pick from generated parents."""
        ordered: list[TableSpec] = []
        placed: set[str] = set()
        remaining = list(self.tables)
        while remaining:
            ready = [table for table in remaining if parents_of(table) <= placed]
            if not ready:
                stuck = ", ".join(table.name for table in remaining)
                raise ValueError(f"foreign keys form a cycle among: {stuck}")
            for table in ready:
                ordered.append(table)
                placed.add(table.name)
                remaining.remove(table)
        return ordered


def parents_of(table: TableSpec) -> set[str]:
    parents: set[str] = set()
    for column in table.columns:
        if column.references is not None:
            parents.add(column.references.split(".")[0])
    # A self-reference is filled in after the table's own keys, so it is no dependency.
    parents.discard(table.name)
    return parents


def check_reference(schema: SchemaSpec, table: TableSpec, column: ColumnSpec, target: str) -> None:
    where = f"{table.name}.{column.name}"
    parts = target.split(".")
    if len(parts) != 2:
        raise ValueError(f"{where}: references must look like table.column")
    target_table_name, target_column_name = parts
    # A hierarchy (employee -> manager) needs a NULL parent for its root.
    if target_table_name == table.name and not column.nullable:
        raise ValueError(f"{where}: self-references need nullable: true for the root")
    try:
        target_table = schema.table(target_table_name)
    except KeyError:
        raise ValueError(f"{where}: unknown table {target_table_name}") from None
    if target_table.primary_key != target_column_name:
        raise ValueError(f"{where}: {target} is not the primary key of {target_table_name}")
    target_kind = target_table.column(target_column_name).logical_type.kind
    if target_kind != column.logical_type.kind:
        raise ValueError(f"{where}: type differs from {target}")


def load_schema(path: Path) -> SchemaSpec:
    """Load and validate one schema file. Raises ValueError or pydantic.ValidationError."""
    with path.open() as handle:
        raw = yaml.safe_load(handle)
    schema = SchemaSpec.model_validate(raw)
    if schema.name != path.stem:
        raise ValueError(f"{path.name}: schema name {schema.name!r} must match the file name")
    return schema
