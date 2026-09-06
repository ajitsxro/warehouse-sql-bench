"""Map ANSI SQLSTATE codes to error categories.

Snowflake and Databricks both report a SQLSTATE, and it is the one
classification signal the two share. The first two characters are the class.
"""

from sqlbench.warehouses.base import ErrorCategory

INSUFFICIENT_PRIVILEGE = "42501"


def category_from_sqlstate(sqlstate: str | None) -> ErrorCategory:
    if sqlstate is None or len(sqlstate) != 5:
        return ErrorCategory.OTHER
    sqlstate_class = sqlstate[:2]
    # 42501 sits inside class 42, so check it before the class-wide rule.
    if sqlstate == INSUFFICIENT_PRIVILEGE or sqlstate_class == "28":
        return ErrorCategory.PERMISSION
    # Class 42 is "syntax error or access rule violation": parse errors plus
    # compile errors such as unknown tables (42S02, 42P01) or columns (42703).
    if sqlstate_class == "42":
        return ErrorCategory.SYNTAX
    # 22 is data exceptions (bad cast, division by zero); 21 is cardinality
    # violations (a scalar subquery returning several rows).
    if sqlstate_class in ("21", "22"):
        return ErrorCategory.RUNTIME
    # 57 is operator intervention; the harness only cancels on timeout.
    if sqlstate_class == "57":
        return ErrorCategory.TIMEOUT
    return ErrorCategory.OTHER
