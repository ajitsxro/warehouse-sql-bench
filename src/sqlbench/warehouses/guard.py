"""Read-only check that runs before any SQL reaches a warehouse."""

import sqlglot
from sqlglot import exp
from sqlglot.errors import SqlglotError

from sqlbench.warehouses.base import ErrorCategory, QueryError, WarehouseName

ALLOWED_ROOTS = (exp.Select, exp.SetOperation, exp.Subquery)

# Checked anywhere in the tree, not just at the root, because some dialects
# accept DML inside a CTE (DuckDB: WITH d AS (DELETE ... RETURNING *) SELECT ...).
FORBIDDEN_NODES = (
    exp.DML,
    exp.DDL,
    exp.Drop,
    exp.Alter,
    exp.TruncateTable,
    exp.Command,
    exp.Into,
)


def check_read_only(sql: str, dialect: WarehouseName) -> QueryError | None:
    """Return an error if sql is not exactly one SELECT or WITH query, else None.

    SQL that sqlglot cannot parse is rejected, since it cannot be shown to be
    safe. That failure is reported as a syntax error.
    """
    try:
        parsed = sqlglot.parse(sql, read=dialect.value)
    except SqlglotError as error:
        return QueryError(ErrorCategory.SYNTAX, f"read-only guard: could not parse SQL: {error}")

    statements = [statement for statement in parsed if statement is not None]
    if len(statements) != 1:
        return QueryError(
            ErrorCategory.PERMISSION,
            f"read-only guard: expected one statement, found {len(statements)}",
        )

    statement = statements[0]
    if not isinstance(statement, ALLOWED_ROOTS):
        return QueryError(
            ErrorCategory.PERMISSION,
            f"read-only guard: only SELECT or WITH queries may run, got {statement.key.upper()}",
        )

    for node in statement.walk():
        if isinstance(node, FORBIDDEN_NODES):
            return QueryError(
                ErrorCategory.PERMISSION,
                f"read-only guard: query contains a {node.key.upper()} clause",
            )
    return None
