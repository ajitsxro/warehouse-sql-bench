"""Rough upper bounds on what warehouse work costs, printed before it runs.

The per-table and per-query times are deliberately high guesses for the
smallest warehouse size; a real load takes a few seconds per table. Prices
are list prices and vary by region and contract.
"""

from sqlbench.warehouses.base import WarehouseName

SECONDS_PER_TABLE = 20
SECONDS_PER_QUERY = 10
# A warehouse keeps billing until it suspends after the last statement.
SUSPEND_AFTER_S = 60
SNOWFLAKE_XSMALL_CREDITS_PER_HOUR = 1.0
SNOWFLAKE_USD_PER_CREDIT = 3.0
DATABRICKS_2XSMALL_DBU_PER_HOUR = 4.0
DATABRICKS_USD_PER_DBU = 0.70


def warehouse_cost(warehouse: WarehouseName, work_s: float, what: str) -> str:
    hours = (work_s + SUSPEND_AFTER_S) / 3600
    if warehouse == WarehouseName.SNOWFLAKE:
        credits = hours * SNOWFLAKE_XSMALL_CREDITS_PER_HOUR
        dollars = credits * SNOWFLAKE_USD_PER_CREDIT
        return (
            f"{what}, at most ~{hours * 60:.0f} min on an X-Small warehouse: "
            f"~{credits:.2f} credits (~${dollars:.2f} at ${SNOWFLAKE_USD_PER_CREDIT:.0f}/credit; "
            "trial credits are free)."
        )
    dbus = hours * DATABRICKS_2XSMALL_DBU_PER_HOUR
    dollars = dbus * DATABRICKS_USD_PER_DBU
    return (
        f"{what}, at most ~{hours * 60:.0f} min on a 2X-Small SQL warehouse: "
        f"~{dbus:.2f} DBU (~${dollars:.2f} at ${DATABRICKS_USD_PER_DBU:.2f}/DBU; "
        "no charge on Free Edition, which has a daily quota instead)."
    )


def cost_estimate(warehouse: WarehouseName, table_count: int) -> str:
    """Loading or checking table_count tables."""
    text = warehouse_cost(warehouse, table_count * SECONDS_PER_TABLE, f"{table_count} tables")
    if warehouse == WarehouseName.SNOWFLAKE:
        text += " Storage is under 1 GB."
    return text


def query_cost_estimate(warehouse: WarehouseName, query_count: int) -> str:
    """Running query_count gold or model queries."""
    return warehouse_cost(warehouse, query_count * SECONDS_PER_QUERY, f"{query_count} queries")
