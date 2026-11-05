import pytest

from sqlbench.warehouses.base import ErrorCategory, WarehouseName
from sqlbench.warehouses.guard import PARSE_FAILURE_PREFIX, check_read_only

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
    ],
)
def test_read_only_queries_pass(sql: str, dialect: WarehouseName) -> None:
    assert check_read_only(sql, dialect) is None


# Regression set: dialect features that gold SQL and model output will use.
# If a sqlglot upgrade stops parsing one of these, the guard would start
# rejecting valid queries, so these must keep passing.
DIALECT_QUERIES = {
    SNOWFLAKE: [
        "SELECT o.id, f.value:sku::STRING AS sku, f.value:qty::INT AS qty"
        " FROM orders o, LATERAL FLATTEN(input => o.payload:items) f",
        "SELECT o.payload:customer.address.city::STRING AS city FROM orders o"
        " WHERE o.payload:status::STRING = 'shipped'",
        "SELECT customer_id, order_id FROM orders"
        " QUALIFY ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY placed_at DESC) = 1",
        "SELECT TRANSFORM(prices, p -> p * 1.1) AS marked_up FROM products",
        "SELECT REDUCE(quantities, 0, (acc, q) -> acc + q) AS total_qty FROM orders",
        "SELECT * FROM monthly_sales PIVOT (SUM(amount) FOR month IN ('JAN', 'FEB', 'MAR')) AS p",
        "SELECT * FROM t MATCH_RECOGNIZE (ORDER BY a PATTERN (up+) DEFINE up AS a > 0)",
    ],
    DATABRICKS: [
        "SELECT o.id, item FROM orders o LATERAL VIEW explode(o.items) exploded AS item",
        "SELECT o.id, e.key, e.value FROM orders o, LATERAL variant_explode(o.payload) AS e",
        "SELECT o.payload:customer.address.city::string AS city FROM orders o",
        "SELECT transform(prices, p -> p * 1.1), filter(prices, p -> p > 10),"
        " aggregate(quantities, 0, (acc, q) -> acc + q) FROM products",
        "SELECT attributes['color'] AS color, shipping.city AS city FROM products",
        "SELECT customer_id, order_id FROM orders"
        " QUALIFY ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY placed_at DESC) = 1",
        "SELECT * FROM monthly_sales PIVOT (SUM(amount) FOR month IN ('JAN', 'FEB', 'MAR'))",
    ],
}

DIALECT_CASES = []
for case_dialect, queries in DIALECT_QUERIES.items():
    for query in queries:
        DIALECT_CASES.append(pytest.param(case_dialect, query, id=case_dialect.value))


@pytest.mark.parametrize(("dialect", "sql"), DIALECT_CASES)
def test_dialect_specific_queries_pass(dialect: WarehouseName, sql: str) -> None:
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
def test_writes_and_non_queries_are_blocked(sql: str, dialect: WarehouseName) -> None:
    error = check_read_only(sql, dialect)
    assert error is not None
    assert error.category == ErrorCategory.BLOCKED
    assert error.message.startswith("read-only guard:")
    assert not error.message.startswith(PARSE_FAILURE_PREFIX)


@pytest.mark.parametrize("sql", ["SELEC 1", "SELECT 'unterminated", "SELECT (1"])
def test_unparseable_sql_is_rejected_as_syntax(sql: str) -> None:
    error = check_read_only(sql, DUCKDB)
    assert error is not None
    assert error.category == ErrorCategory.SYNTAX
    assert error.message.startswith("read-only guard: could not parse")
