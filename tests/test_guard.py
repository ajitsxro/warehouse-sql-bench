import pytest

from sqlbench.warehouses.base import ErrorCategory, WarehouseName
from sqlbench.warehouses.guard import check_read_only

SNOWFLAKE = WarehouseName.SNOWFLAKE
DATABRICKS = WarehouseName.DATABRICKS
DUCKDB = WarehouseName.DUCKDB


@pytest.mark.parametrize(
    ("sql", "dialect"),
    [
        ("SELECT 1", DUCKDB),
        ("SELECT 1;", DUCKDB),
        ("-- leading comment\nSELECT 1", DUCKDB),
        ("(SELECT 1)", DUCKDB),
        ("WITH a AS (SELECT 1 AS x) SELECT x FROM a", DUCKDB),
        ("SELECT 1 UNION ALL SELECT 2", DUCKDB),
        ("SELECT a FROM t WHERE a IN (SELECT b FROM u)", DUCKDB),
        ("SELECT f.value FROM t, LATERAL FLATTEN(input => t.v) f", SNOWFLAKE),
        ("SELECT a FROM t QUALIFY ROW_NUMBER() OVER (ORDER BY a) = 1", SNOWFLAKE),
        ("SELECT v:customer.name::STRING FROM t", SNOWFLAKE),
        (
            "SELECT * FROM t MATCH_RECOGNIZE (ORDER BY a PATTERN (up+) DEFINE up AS a > 0)",
            SNOWFLAKE,
        ),
        ("SELECT a, e FROM t LATERAL VIEW explode(arr) x AS e", DATABRICKS),
        ("SELECT transform(arr, x -> x + 1), m['k'], s.field FROM t", DATABRICKS),
    ],
)
def test_read_only_queries_pass(sql: str, dialect: WarehouseName) -> None:
    assert check_read_only(sql, dialect) is None


@pytest.mark.parametrize(
    ("sql", "dialect"),
    [
        ("INSERT INTO t VALUES (1)", DUCKDB),
        ("UPDATE t SET a = 1", DUCKDB),
        ("DELETE FROM t", DUCKDB),
        ("MERGE INTO t USING s ON t.a = s.a WHEN MATCHED THEN DELETE", SNOWFLAKE),
        ("CREATE TABLE t AS SELECT 1", DATABRICKS),
        ("DROP TABLE t", DUCKDB),
        ("ALTER TABLE t ADD COLUMN b INT", DUCKDB),
        ("TRUNCATE TABLE t", SNOWFLAKE),
        ("GRANT SELECT ON t TO ROLE r", SNOWFLAKE),
        ("CALL some_procedure()", SNOWFLAKE),
        ("SHOW TABLES", SNOWFLAKE),
        ("USE DATABASE other", SNOWFLAKE),
        ("SET x = 1", SNOWFLAKE),
        ("COPY INTO t FROM @stage", SNOWFLAKE),
        ("INSERT OVERWRITE t SELECT 1", DATABRICKS),
        ("OPTIMIZE t", DATABRICKS),
        # The root is a query but it still writes.
        ("WITH a AS (SELECT 1) INSERT INTO t SELECT * FROM a", DATABRICKS),
        ("SELECT * INTO t2 FROM t", DUCKDB),
        ("WITH d AS (DELETE FROM t RETURNING *) SELECT * FROM d", DUCKDB),
        # A harmless first statement must not let a second one through.
        ("SELECT 1; DROP TABLE t", DUCKDB),
        ("SELECT 1; SELECT 2", DUCKDB),
        ("", DUCKDB),
        ("-- only a comment", DUCKDB),
    ],
)
def test_writes_and_non_queries_are_rejected_as_permission(
    sql: str, dialect: WarehouseName
) -> None:
    error = check_read_only(sql, dialect)
    assert error is not None
    assert error.category == ErrorCategory.PERMISSION
    assert error.message.startswith("read-only guard:")


@pytest.mark.parametrize("sql", ["SELEC 1", "SELECT 'unterminated", "SELECT (1"])
def test_unparseable_sql_is_rejected_as_syntax(sql: str) -> None:
    error = check_read_only(sql, DUCKDB)
    assert error is not None
    assert error.category == ErrorCategory.SYNTAX
