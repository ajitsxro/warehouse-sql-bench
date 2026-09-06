import pytest
from snowflake.connector.errors import DatabaseError, ProgrammingError

from sqlbench.warehouses.base import ErrorCategory
from sqlbench.warehouses.snowflake_warehouse import categorize_snowflake_error
from sqlbench.warehouses.sqlstate import category_from_sqlstate


@pytest.mark.parametrize(
    ("sqlstate", "category"),
    [
        ("42000", ErrorCategory.SYNTAX),
        ("42601", ErrorCategory.SYNTAX),
        ("42S02", ErrorCategory.SYNTAX),
        ("42P01", ErrorCategory.SYNTAX),
        ("42501", ErrorCategory.PERMISSION),
        ("28000", ErrorCategory.PERMISSION),
        ("22012", ErrorCategory.RUNTIME),
        ("21000", ErrorCategory.RUNTIME),
        ("57014", ErrorCategory.TIMEOUT),
        ("08001", ErrorCategory.OTHER),
        ("XX000", ErrorCategory.OTHER),
        (None, ErrorCategory.OTHER),
        ("", ErrorCategory.OTHER),
        ("42", ErrorCategory.OTHER),
    ],
)
def test_category_from_sqlstate(sqlstate: str | None, category: ErrorCategory) -> None:
    assert category_from_sqlstate(sqlstate) == category


@pytest.mark.parametrize(
    ("errno", "sqlstate", "message", "category"),
    [
        (1003, "42000", "SQL compilation error: syntax error line 1 at position 0", "syntax"),
        (
            904,
            "42000",
            "SQL compilation error: error line 1 at position 7 invalid identifier",
            "syntax",
        ),
        (2003, "42S02", "Object 'MISSING' does not exist or not authorized.", "syntax"),
        (100038, "22018", "Numeric value 'abc' is not recognized", "runtime"),
        (100051, "22012", "Division by zero", "runtime"),
        (630, "57014", "Statement reached its statement or warehouse timeout", "timeout"),
        (3001, "42501", "Insufficient privileges to operate on schema", "permission"),
    ],
)
def test_snowflake_programming_errors(
    errno: int, sqlstate: str, message: str, category: str
) -> None:
    error = ProgrammingError(msg=message, errno=errno, sqlstate=sqlstate)
    assert categorize_snowflake_error(error) == ErrorCategory(category)


def test_snowflake_cancel_is_timeout_even_without_sqlstate() -> None:
    error = ProgrammingError(msg="SQL execution canceled", errno=604)
    assert categorize_snowflake_error(error) == ErrorCategory.TIMEOUT


def test_snowflake_connection_failure_is_other() -> None:
    error = DatabaseError(msg="Failed to connect to DB", errno=250001, sqlstate="08001")
    assert categorize_snowflake_error(error) == ErrorCategory.OTHER
