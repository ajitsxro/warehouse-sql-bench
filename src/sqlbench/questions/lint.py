"""Static checks on question files: format, schema references, and gold SQL rules.

Nothing here touches a warehouse. Table and column names are resolved with
sqlglot against the schema spec, in each gold's own dialect.
"""

from collections.abc import Callable
from pathlib import Path

import sqlglot
import yaml
from sqlglot import exp
from sqlglot.errors import SqlglotError
from sqlglot.optimizer.qualify import qualify

from sqlbench.questions.model import Question, load_question
from sqlbench.schema.ddl import databricks_type, snowflake_type
from sqlbench.schema.spec import SchemaSpec, load_schema
from sqlbench.schema.types import LogicalType
from sqlbench.warehouses.base import WarehouseName
from sqlbench.warehouses.guard import check_read_only

GOLD_DIALECTS: dict[WarehouseName, Callable[[LogicalType], str]] = {
    WarehouseName.SNOWFLAKE: snowflake_type,
    WarehouseName.DATABRICKS: databricks_type,
}


def lint_files(paths: list[Path], schemas_dir: Path) -> dict[Path, list[str]]:
    """Problems per question file; a file with an empty list passed."""
    problems: dict[Path, list[str]] = {}
    first_path_by_id: dict[str, Path] = {}
    schemas: dict[str, SchemaSpec] = {}
    for path in paths:
        try:
            question = load_question(path)
        except (OSError, ValueError, yaml.YAMLError) as error:
            # pydantic's ValidationError is a ValueError and lists every field problem.
            problems[path] = [f"invalid: {error}"]
            continue

        found: list[str] = []
        if path.stem != question.id:
            found.append(f"file name must be {question.id}.yaml")
        if path.parent.name != question.schema_name:
            found.append(f"file must be in questions/{question.schema_name}/")
        if question.id in first_path_by_id:
            found.append(f"id {question.id} is also used by {first_path_by_id[question.id]}")
        else:
            first_path_by_id[question.id] = path

        schema_path = schemas_dir / f"{question.schema_name}.yaml"
        if question.schema_name not in schemas and schema_path.exists():
            schemas[question.schema_name] = load_schema(schema_path)
        if question.schema_name in schemas:
            found.extend(lint_gold_sql(question, schemas[question.schema_name]))
        else:
            found.append(f"unknown schema {question.schema_name}")
        problems[path] = found
    return problems


def lint_gold_sql(question: Question, schema: SchemaSpec) -> list[str]:
    problems: list[str] = []
    for warehouse, type_name in GOLD_DIALECTS.items():
        label = f"{warehouse} gold"
        sql = question.gold_sql.for_warehouse(warehouse)
        guard_error = check_read_only(sql, warehouse)
        if guard_error is not None:
            problems.append(f"{label}: {guard_error.message}")
            continue
        tree = sqlglot.parse_one(sql, read=warehouse.value)

        found = table_problems(tree, schema)
        if not found:
            # Column resolution would only repeat an unknown-table problem.
            found = column_problems(tree, schema, warehouse, type_name)
        found.extend(limit_problems(tree, schema))
        if question.order_matters and tree.args.get("order") is None:
            found.append("order_matters is true but the query has no outer ORDER BY")
        for problem in found:
            problems.append(f"{label}: {problem}")
    return problems


def table_problems(tree: exp.Expr, schema: SchemaSpec) -> list[str]:
    known = {table.name for table in schema.tables}
    cte_names = {cte.alias_or_name.lower() for cte in tree.find_all(exp.CTE)}
    problems: list[str] = []
    for table in tree.find_all(exp.Table):
        # Table functions such as TABLE(FLATTEN(...)) have no plain name.
        if not isinstance(table.this, exp.Identifier):
            continue
        name = table.name.lower()
        if table.args.get("db") is not None:
            # The session schema picks main or edge; a qualified name would pin one.
            problems.append(f"{table.sql()}: use the bare table name")
        elif name not in known and name not in cte_names:
            problems.append(f"unknown table {table.name}")
    return problems


def column_problems(
    tree: exp.Expr,
    schema: SchemaSpec,
    warehouse: WarehouseName,
    type_name: Callable[[LogicalType], str],
) -> list[str]:
    mapping: dict[str, object] = {}
    for table in schema.tables:
        columns = {}
        for column in table.columns:
            columns[column.name] = type_name(column.logical_type)
        mapping[table.name] = columns
    try:
        qualify(tree.copy(), schema=mapping, dialect=warehouse.value, validate_qualify_columns=True)
    except SqlglotError as error:
        return [f"column check failed: {error}"]
    return []


def limit_problems(tree: exp.Expr, schema: SchemaSpec) -> list[str]:
    """LIMIT is only deterministic if the ORDER BY ends in a unique key.

    A key here is a primary key column of a schema table, or the full GROUP BY
    of the same SELECT. It's a name check: a join that repeats a key is not
    caught, which is what the edge dataset's duplicates are for.
    """
    primary_keys = set()
    for table in schema.tables:
        if table.primary_key is not None:
            primary_keys.add(table.primary_key)
    problems: list[str] = []
    for node in tree.find_all(exp.Select, exp.SetOperation):
        if node.args.get("limit") is None:
            continue
        if not ends_in_unique_key(node, primary_keys):
            problems.append(
                "LIMIT needs an ORDER BY that ends in a unique key (a primary key or "
                "every GROUP BY column); for top N with ties use RANK() in QUALIFY "
                "or a subquery, and say so in the question"
            )
    return problems


def ends_in_unique_key(node: exp.Expr, primary_keys: set[str]) -> bool:
    order = node.args.get("order")
    if order is None:
        return False
    ordered = [canonical(term.this, node) for term in order.expressions]
    last = ordered[-1]
    if isinstance(last, exp.Column) and last.name.lower() in primary_keys:
        return True
    group = node.args.get("group")
    if group is None:
        return False
    ordered_sql = {expression.sql().lower() for expression in ordered}
    for expression in group.expressions:
        if canonical(expression, node).sql().lower() not in ordered_sql:
            return False
    return True


def canonical(expression: exp.Expr, node: exp.Expr) -> exp.Expr:
    """Resolve a position (ORDER BY 2) or an output alias to the selected expression."""
    if not isinstance(node, exp.Select):
        return expression
    projections: list[exp.Expr] = node.expressions
    if isinstance(expression, exp.Literal) and expression.is_int:
        position = int(expression.this)
        if 1 <= position <= len(projections):
            return projections[position - 1].unalias()
    if isinstance(expression, exp.Column) and not expression.table:
        for projection in projections:
            if isinstance(projection, exp.Alias) and projection.alias == expression.name:
                return projection.unalias()
    return expression
