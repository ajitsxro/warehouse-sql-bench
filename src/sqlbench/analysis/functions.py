"""Built-in function names of each warehouse, fetched once and committed.

The dialect_leak label asks "does this function exist on the other
warehouse?". Answering from the warehouses' own catalogs, rather than a
hand-written list, keeps the label honest. The lists are refreshed with
`bench analysis functions`.
"""

import datetime
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from sqlbench.warehouses.base import WarehouseName

FUNCTIONS_DIR = Path(__file__).parent / "functions"
NAME = re.compile(r"^[a-z_][a-z0-9_]*$")

# SHOW FUNCTIONS on Snowflake leaves out system table functions, so these
# come from the documentation's table-function list instead.
SNOWFLAKE_TABLE_FUNCTIONS = frozenset(
    {"flatten", "split_to_table", "strtok_split_to_table", "generator", "result_scan"}
)

# Snowflake: SHOW FUNCTIONS lists built-ins (is_builtin = 'Y') by name in
# column 1. Databricks: SHOW SYSTEM FUNCTIONS has one column of names.
LISTING_SQL = {
    WarehouseName.SNOWFLAKE: "SHOW FUNCTIONS",
    WarehouseName.DATABRICKS: "SHOW SYSTEM FUNCTIONS",
}


def names_from_listing(warehouse: WarehouseName, rows: list[tuple[Any, ...]]) -> list[str]:
    """Lower-case function names from a listing, without operators like '!='."""
    names = set()
    for row in rows:
        if warehouse == WarehouseName.SNOWFLAKE:
            if row[3] != "Y":
                continue
            name = str(row[1]).lower()
        else:
            name = str(row[0]).lower()
        if NAME.match(name):
            names.add(name)
    if warehouse == WarehouseName.SNOWFLAKE:
        names |= SNOWFLAKE_TABLE_FUNCTIONS
    return sorted(names)


def fetch_function_names(warehouse: WarehouseName, connection: Any) -> list[str]:
    cursor = connection.cursor()
    try:
        cursor.execute(LISTING_SQL[warehouse])
        rows = cursor.fetchall()
    finally:
        cursor.close()
    return names_from_listing(warehouse, [tuple(row) for row in rows])


def function_list_path(warehouse: WarehouseName, directory: Path = FUNCTIONS_DIR) -> Path:
    return directory / f"{warehouse.value}.txt"


def write_function_list(
    warehouse: WarehouseName, names: list[str], fetched_on: datetime.date, directory: Path
) -> Path:
    path = function_list_path(warehouse, directory)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = f"# {warehouse.value} built-in functions, {LISTING_SQL[warehouse]}, {fetched_on}\n"
    path.write_text(header + "\n".join(names) + "\n")
    return path


def load_function_names(
    warehouse: WarehouseName, directory: Path = FUNCTIONS_DIR
) -> frozenset[str]:
    names = set()
    for line in function_list_path(warehouse, directory).read_text().splitlines():
        if line and not line.startswith("#"):
            names.add(line)
    return frozenset(names)


def fetch_and_write(
    connect: Callable[[WarehouseName], Any], directory: Path, today: datetime.date
) -> dict[WarehouseName, int]:
    """Fetch both warehouses' lists; returns how many names each has."""
    counts = {}
    for warehouse in (WarehouseName.SNOWFLAKE, WarehouseName.DATABRICKS):
        connection = connect(warehouse)
        try:
            names = fetch_function_names(warehouse, connection)
        finally:
            connection.close()
        write_function_list(warehouse, names, today, directory)
        counts[warehouse] = len(names)
    return counts
