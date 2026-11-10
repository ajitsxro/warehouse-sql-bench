"""SQL extraction from messy, realistic model responses."""

import pytest

from sqlbench.models.extract import extract_sql, parses_as_sql
from sqlbench.warehouses.base import WarehouseName

SNOWFLAKE = WarehouseName.SNOWFLAKE
DATABRICKS = WarehouseName.DATABRICKS


def test_prose_before_and_after_the_block() -> None:
    response = """Sure! To count orders per status, group by status:

```sql
SELECT status, COUNT(*) AS orders
FROM orders
GROUP BY status;
```

This returns one row per status. Let me know if you need anything else."""

    assert extract_sql(response, SNOWFLAKE) == (
        "SELECT status, COUNT(*) AS orders\nFROM orders\nGROUP BY status"
    )


def test_the_first_sql_block_wins() -> None:
    response = """First attempt:
```sql
SELECT 1
```
Actually, a better version:
```sql
SELECT 2
```"""

    assert extract_sql(response, SNOWFLAKE) == "SELECT 1"


def test_other_blocks_before_the_sql_block_are_skipped() -> None:
    response = """The schema is:
```python
print("not sql")
```
```sql
SELECT 3
```"""

    assert extract_sql(response, SNOWFLAKE) == "SELECT 3"


@pytest.mark.parametrize("fence", ["```SQL", "```Sql", "``` sql", "```sql title=query"])
def test_fence_variants(fence: str) -> None:
    assert extract_sql(f"{fence}\nSELECT 1\n```", SNOWFLAKE) == "SELECT 1"


def test_other_languages_named_like_sql_are_not_sql_blocks() -> None:
    response = "```sqlite\nSELECT 1\n```"

    assert extract_sql(response, SNOWFLAKE) is None


@pytest.mark.parametrize(
    ("block", "expected"),
    [
        ("SELECT 1;", "SELECT 1"),
        ("SELECT 1;;  \n", "SELECT 1"),
        ("SELECT 1 ;\n;\n", "SELECT 1"),
        ("\n\n  SELECT ';' AS semicolon;\n", "SELECT ';' AS semicolon"),
    ],
)
def test_trailing_semicolons_and_whitespace_are_removed(block: str, expected: str) -> None:
    assert extract_sql(f"```sql\n{block}```", SNOWFLAKE) == expected


def test_windows_line_endings() -> None:
    assert extract_sql("Here:\r\n```sql\r\nSELECT 1\r\n```\r\n", SNOWFLAKE) == "SELECT 1"


def test_an_unclosed_block_runs_to_the_end() -> None:
    response = "```sql\nSELECT customer_id, COUNT(*)\nFROM orders\nGROUP BY customer_id"

    assert extract_sql(response, SNOWFLAKE) == (
        "SELECT customer_id, COUNT(*)\nFROM orders\nGROUP BY customer_id"
    )


def test_an_empty_block_is_no_sql() -> None:
    assert extract_sql("```sql\n\n```", SNOWFLAKE) is None
    assert extract_sql("```sql\n;\n```", SNOWFLAKE) is None


def test_a_bare_sql_response_is_used_whole() -> None:
    response = "SELECT status, COUNT(*) FROM orders GROUP BY status;\n"

    assert extract_sql(response, SNOWFLAKE) == "SELECT status, COUNT(*) FROM orders GROUP BY status"


def test_bare_sql_is_parsed_in_the_warehouse_dialect() -> None:
    response = "SELECT tag FROM products LATERAL VIEW explode(tags) t AS tag"

    assert extract_sql(response, DATABRICKS) == response


def test_a_bare_write_is_kept_for_the_guard_to_block() -> None:
    assert extract_sql("DELETE FROM orders", SNOWFLAKE) == "DELETE FROM orders"


@pytest.mark.parametrize(
    "response",
    [
        "I can't answer that without more information about the schema.",
        "Hello",
        "The answer is 42.",
        "",
        "   \n",
        "```\nSELECT 1\n```\nThat's the query.",
    ],
)
def test_responses_without_sql(response: str) -> None:
    assert extract_sql(response, SNOWFLAKE) is None


def test_reasoning_drafts_are_ignored() -> None:
    response = """<think>
Maybe:
```sql
SELECT wrong
```
No, I need the count.
</think>
```sql
SELECT COUNT(*) FROM orders
```"""

    assert extract_sql(response, SNOWFLAKE) == "SELECT COUNT(*) FROM orders"


def test_reasoning_cut_off_before_an_answer_is_no_sql() -> None:
    response = "<think>\nLet me draft:\n```sql\nSELECT 1\n```\nand then"

    assert extract_sql(response, SNOWFLAKE) is None


@pytest.mark.parametrize(
    ("text", "parses"),
    [
        ("SELECT 1", True),
        ("WITH a AS (SELECT 1 AS x) SELECT x FROM a", True),
        ("SELECT 1; SELECT 2", True),
        ("Hello world", False),
        ("SELECT FROM WHERE", False),
    ],
)
def test_parses_as_sql(text: str, parses: bool) -> None:
    assert parses_as_sql(text, SNOWFLAKE) is parses
