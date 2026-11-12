"""Automatic failure labels, computed from a stored row, its question and its schema.

Labels describe why a prediction failed. They never change a score. One
failure can carry several labels; a correct row carries none.

- dialect_leak: the warehouse rejected the SQL, and it uses the other
  warehouse's function or parses only as the other warehouse's dialect.
- unknown_function: the warehouse rejected a function neither warehouse has
  (for example SQLite's STRFTIME).
- schema_hallucination: the warehouse rejected a table or column name that
  isn't in the schema and isn't defined by the query itself.
- semantics_trap: a semantics question answered wrongly with SQL that
  contains that question's trap (for example DATEDIFF for a boundary trap).
- lucky_match: correct on main, wrong on edge.
- near-miss tags from the scorer, as stored ("edge:row_subset").
"""

import re
from dataclasses import dataclass
from enum import StrEnum

import sqlglot
from sqlglot import Token, TokenType, exp
from sqlglot.errors import SqlglotError

from sqlbench.analysis.functions import SNOWFLAKE_TABLE_FUNCTIONS
from sqlbench.harness.store import ResultRow
from sqlbench.questions.model import Divergence, Question
from sqlbench.schema.spec import SchemaSpec
from sqlbench.warehouses.base import WarehouseName


class Label(StrEnum):
    DIALECT_LEAK = "dialect_leak"
    UNKNOWN_FUNCTION = "unknown_function"
    SCHEMA_HALLUCINATION = "schema_hallucination"
    SEMANTICS_TRAP = "semantics_trap"
    LUCKY_MATCH = "lucky_match"


class Trap(StrEnum):
    NULL_ORDER = "null_order"
    DATEDIFF_BOUNDARY = "datediff_boundary"
    GREATEST_LEAST_NULL = "greatest_least_null"
    DAY_OF_WEEK = "day_of_week"
    ADD_MONTHS_END = "add_months_end"
    COUNT_VARIANT = "count_variant"
    CAST_TO_INT = "cast_to_int"


# Semantics questions name the probe that confirmed their trap in
# divergence_note, e.g. "(probe datediff_year)"; docs/semantics_probes.md.
TRAP_BY_PROBE = {
    "null_order_desc": Trap.NULL_ORDER,
    "null_order_asc": Trap.NULL_ORDER,
    "datediff_day_ts": Trap.DATEDIFF_BOUNDARY,
    "datediff_year": Trap.DATEDIFF_BOUNDARY,
    "datediff_month": Trap.DATEDIFF_BOUNDARY,
    "datediff_month_full": Trap.DATEDIFF_BOUNDARY,
    "greatest_null": Trap.GREATEST_LEAST_NULL,
    "dayofweek": Trap.DAY_OF_WEEK,
    "add_months_feb29": Trap.ADD_MONTHS_END,
    "count_variant_jsonnull": Trap.COUNT_VARIANT,
    "count_attributes_edge": Trap.COUNT_VARIANT,
    "cast_decimal_int": Trap.CAST_TO_INT,
}
PROBE_NAMES = re.compile(r"\(probes? ([a-z0-9_, ]+)\)")

ERROR_OUTCOMES = {"syntax_error", "runtime_error"}
WRONG_ANSWER_OUTCOMES = {"wrong_result", "empty_result", "wrong_columns"}

# The function names a warehouse says it can't resolve.
UNKNOWN_FUNCTION_PATTERNS = [
    re.compile(r"Unknown (?:user-defined )?functions? ([A-Za-z0-9_, ]+?)(?: \(|\.|$)"),
    re.compile(r"Cannot resolve routine `([A-Za-z0-9_]+)`"),
    re.compile(r"Could not resolve ([A-Za-z0-9_]+) to a table-valued function"),
]
# The identifier a warehouse says is not a table or column.
UNKNOWN_COLUMN_PATTERNS = [
    re.compile(r"invalid identifier '([^']+)'"),
    re.compile(r"with name (`[^ ]+`) cannot be resolved"),
]
UNKNOWN_TABLE_PATTERNS = [
    re.compile(r"Object '([^']+)' does not exist"),
    re.compile(r"The table or view (`[^ ]+`) cannot be found"),
]
# Columns that table functions produce without being declared:
# Snowflake FLATTEN, Databricks explode/posexplode.
GENERATED_COLUMNS = frozenset({"seq", "key", "path", "index", "value", "this", "col", "pos"})
DAY_OF_WEEK_WORDS = frozenset({"dayofweek", "dayofweek_iso", "dayofweekiso", "dow", "dow_iso"})
DATEDIFF_FUNCTIONS = frozenset({"datediff", "date_diff", "timestampdiff", "timestamp_diff"})
DATEADD_FUNCTIONS = frozenset({"dateadd", "date_add", "timestampadd", "timestamp_add"})
INTEGER_TYPES = frozenset({TokenType.INT, TokenType.BIGINT, TokenType.SMALLINT, TokenType.TINYINT})


@dataclass(frozen=True)
class SchemaNames:
    """Lower-case names the schema defines, and its semi-structured columns."""

    tables: frozenset[str]
    columns: frozenset[str]
    semi_structured: frozenset[str]

    @classmethod
    def from_spec(cls, spec: SchemaSpec) -> "SchemaNames":
        tables = set()
        columns = set()
        semi_structured = set()
        for table in spec.tables:
            tables.add(table.name.lower())
            for column in table.columns:
                columns.add(column.name.lower())
                if column.type == "variant" or column.type.startswith("map<"):
                    semi_structured.add(column.name.lower())
        return cls(frozenset(tables), frozenset(columns), frozenset(semi_structured))


def other_warehouse(warehouse: WarehouseName) -> WarehouseName:
    if warehouse == WarehouseName.SNOWFLAKE:
        return WarehouseName.DATABRICKS
    return WarehouseName.SNOWFLAKE


def tokens(sql: str, warehouse: WarehouseName) -> list[Token]:
    try:
        return list(sqlglot.tokenize(sql, read=warehouse.value))
    except SqlglotError:
        return []


def called_functions(sql_tokens: list[Token]) -> set[str]:
    """Lower-case names written as name( ... ), keywords like IN( excluded."""
    names = set()
    for position in range(len(sql_tokens) - 1):
        token = sql_tokens[position]
        following = sql_tokens[position + 1]
        if token.token_type == TokenType.VAR and following.token_type == TokenType.L_PAREN:
            names.add(token.text.lower())
    return names


def words(sql_tokens: list[Token]) -> set[str]:
    return {token.text.lower() for token in sql_tokens}


def unknown_functions(error_message: str) -> set[str]:
    names = set()
    for pattern in UNKNOWN_FUNCTION_PATTERNS:
        for match in pattern.finditer(error_message):
            for name in match.group(1).split(","):
                if name.strip():
                    names.add(name.strip().lower())
    return names


def parses(sql: str, warehouse: WarehouseName) -> bool:
    try:
        sqlglot.parse(sql, read=warehouse.value)
    except SqlglotError:
        return False
    return True


def is_dialect_leak(row: ResultRow, functions: dict[WarehouseName, frozenset[str]]) -> bool:
    if row.outcome not in ERROR_OUTCOMES or not row.predicted_sql:
        return False
    here = WarehouseName(row.warehouse)
    there = other_warehouse(here)
    rejected = unknown_functions(row.error_message or "")
    for name in rejected:
        if name in functions[there]:
            return True
    there_tokens = tokens(row.predicted_sql, there)
    for name in called_functions(there_tokens):
        if name in functions[there] and name not in functions[here]:
            return True
    if here == WarehouseName.DATABRICKS and snowflake_table_function_used(there_tokens):
        return True
    return not parses(row.predicted_sql, here) and parses(row.predicted_sql, there)


def snowflake_table_function_used(sql_tokens: list[Token]) -> bool:
    """LATERAL FLATTEN(...) or TABLE(FLATTEN(...)): Databricks has a scalar flatten,
    so the name alone isn't a leak, but using it as a table is."""
    for position in range(len(sql_tokens) - 2):
        opener = sql_tokens[position]
        name = sql_tokens[position + 1]
        after = sql_tokens[position + 2]
        if after.token_type != TokenType.L_PAREN:
            continue
        if name.text.lower() not in SNOWFLAKE_TABLE_FUNCTIONS:
            continue
        if opener.token_type == TokenType.LATERAL or opener.token_type == TokenType.L_PAREN:
            return True
    return False


def is_unknown_function(row: ResultRow, functions: dict[WarehouseName, frozenset[str]]) -> bool:
    if row.outcome not in ERROR_OUTCOMES:
        return False
    for name in unknown_functions(row.error_message or ""):
        if (
            name not in functions[WarehouseName.SNOWFLAKE]
            and name not in (functions[WarehouseName.DATABRICKS])
        ):
            return True
    return False


def names_defined_by_query(sql: str, warehouse: WarehouseName) -> set[str]:
    """CTE names, table aliases, their column lists, and column aliases."""
    try:
        statements = sqlglot.parse(sql, read=warehouse.value)
    except SqlglotError:
        return set()
    names = set()
    for statement in statements:
        if statement is None:
            continue
        for node in statement.find_all(exp.CTE, exp.Alias):
            if node.alias:
                names.add(node.alias.lower())
        # A TableAlias keeps its name in `this`, not in `alias`.
        for table_alias in statement.find_all(exp.TableAlias):
            if table_alias.name:
                names.add(table_alias.name.lower())
            for column in table_alias.columns:
                names.add(column.name.lower())
    return names


def rejected_names(error_message: str, patterns: list[re.Pattern[str]]) -> list[list[str]]:
    """Each rejected identifier, split into lower-case parts: `r`.`x` -> ['r', 'x']."""
    found = []
    for pattern in patterns:
        for match in pattern.finditer(error_message):
            text = match.group(1).replace("`", "").replace('"', "")
            parts = [part.lower() for part in text.split(".") if part]
            if parts:
                found.append(parts)
    return found


def is_schema_hallucination(row: ResultRow, schema: SchemaNames) -> bool:
    if row.outcome not in ERROR_OUTCOMES or not row.predicted_sql:
        return False
    message = row.error_message or ""
    defined = names_defined_by_query(row.predicted_sql, WarehouseName(row.warehouse))
    # A column name used as a table (FROM tags) is an invented table.
    for parts in rejected_names(message, UNKNOWN_TABLE_PATTERNS):
        if parts[-1] not in schema.tables | defined:
            return True
    known_columns = schema.tables | schema.columns | defined | GENERATED_COLUMNS
    for parts in rejected_names(message, UNKNOWN_COLUMN_PATTERNS):
        # column.field on a real column is a nested-access mistake, not an invented name.
        if len(parts) > 1 and parts[0] in schema.columns:
            continue
        if parts[-1] not in known_columns:
            return True
    return False


def question_traps(question: Question) -> set[Trap]:
    traps = set()
    for match in PROBE_NAMES.finditer(question.divergence_note or ""):
        for probe in match.group(1).split(","):
            trap = TRAP_BY_PROBE.get(probe.strip())
            if trap is not None:
                traps.add(trap)
    return traps


def contains_trap(sql: str, warehouse: WarehouseName, trap: Trap, schema: SchemaNames) -> bool:
    sql_tokens = tokens(sql, warehouse)
    called = called_functions(sql_tokens)
    written = words(sql_tokens)
    if trap == Trap.NULL_ORDER:
        has_order = any(token.token_type == TokenType.ORDER_BY for token in sql_tokens)
        return has_order and "nulls" not in written
    if trap == Trap.DATEDIFF_BOUNDARY:
        return bool(called & DATEDIFF_FUNCTIONS)
    if trap == Trap.GREATEST_LEAST_NULL:
        return bool(called & {"greatest", "least"})
    if trap == Trap.DAY_OF_WEEK:
        return bool(written & DAY_OF_WEEK_WORDS)
    if trap == Trap.ADD_MONTHS_END:
        month_words = written & {"month", "months", "mm", "mon"}
        adds_months = bool(called & DATEADD_FUNCTIONS) and bool(month_words)
        interval_months = "interval" in written and bool(month_words)
        return "add_months" in called or adds_months or interval_months
    if trap == Trap.COUNT_VARIANT:
        return "count" in called and bool(written & schema.semi_structured)
    if trap == Trap.CAST_TO_INT:
        casts = bool(called & {"cast", "try_cast"}) or any(
            token.token_type == TokenType.DCOLON for token in sql_tokens
        )
        return casts and any(token.token_type in INTEGER_TYPES for token in sql_tokens)
    raise ValueError(f"unknown trap {trap}")


def is_semantics_trap(row: ResultRow, question: Question, schema: SchemaNames) -> bool:
    if question.divergence != Divergence.SEMANTICS or row.outcome not in WRONG_ANSWER_OUTCOMES:
        return False
    if not row.predicted_sql:
        return False
    warehouse = WarehouseName(row.warehouse)
    for trap in question_traps(question):
        if contains_trap(row.predicted_sql, warehouse, trap, schema):
            return True
    return False


def auto_labels(
    row: ResultRow,
    question: Question,
    schema: SchemaNames,
    functions: dict[WarehouseName, frozenset[str]],
) -> tuple[str, ...]:
    """Every label that applies to one stored row, labels first, then near-miss tags."""
    if row.outcome == "correct":
        return ()
    labels: list[str] = []
    if is_dialect_leak(row, functions):
        labels.append(Label.DIALECT_LEAK)
    if is_unknown_function(row, functions):
        labels.append(Label.UNKNOWN_FUNCTION)
    if is_schema_hallucination(row, schema):
        labels.append(Label.SCHEMA_HALLUCINATION)
    if is_semantics_trap(row, question, schema):
        labels.append(Label.SEMANTICS_TRAP)
    if row.lucky_match:
        labels.append(Label.LUCKY_MATCH)
    labels.extend(row.near_miss)
    return tuple(labels)
