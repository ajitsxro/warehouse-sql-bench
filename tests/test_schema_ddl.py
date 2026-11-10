from pathlib import Path

import pytest
import sqlglot

from sqlbench.schema.ddl import create_table_statements, databricks_type, snowflake_type
from sqlbench.schema.spec import load_schema
from sqlbench.schema.types import parse_type
from sqlbench.warehouses.base import WarehouseName

SCHEMA = load_schema(Path(__file__).parent / "fixtures" / "mini_shop.yaml")


@pytest.mark.parametrize(
    ("logical", "snowflake", "databricks"),
    [
        ("int", "INTEGER", "INT"),
        ("bigint", "BIGINT", "BIGINT"),
        ("decimal(12,2)", "NUMBER(12,2)", "DECIMAL(12,2)"),
        ("double", "FLOAT", "DOUBLE"),
        ("string", "VARCHAR", "STRING"),
        ("bool", "BOOLEAN", "BOOLEAN"),
        ("date", "DATE", "DATE"),
        ("timestamp_ntz", "TIMESTAMP_NTZ", "TIMESTAMP_NTZ"),
        ("timestamp_tz", "TIMESTAMP_TZ", "TIMESTAMP"),
        ("variant", "VARIANT", "VARIANT"),
        ("array<string>", "ARRAY", "ARRAY<STRING>"),
        ("map<string,decimal(10,2)>", "OBJECT", "MAP<STRING, DECIMAL(10,2)>"),
        (
            "struct<source:string,hits:array<int>>",
            "OBJECT",
            "STRUCT<source: STRING, hits: ARRAY<INT>>",
        ),
    ],
)
def test_type_names(logical: str, snowflake: str, databricks: str) -> None:
    logical_type = parse_type(logical)

    assert snowflake_type(logical_type) == snowflake
    assert databricks_type(logical_type) == databricks


def test_snowflake_orders_table() -> None:
    statements = create_table_statements(SCHEMA, WarehouseName.SNOWFLAKE, "mini_shop_edge")

    assert statements[1] == (
        "CREATE OR REPLACE TABLE mini_shop_edge.orders (\n"
        "    order_id BIGINT NOT NULL,\n"
        "    customer_id BIGINT,\n"
        "    placed_at TIMESTAMP_NTZ NOT NULL,\n"
        "    shipped_at TIMESTAMP_NTZ,\n"
        "    total NUMBER(10,2) NOT NULL,\n"
        "    quantity INTEGER NOT NULL,\n"
        "    weight_kg FLOAT,\n"
        "    attributes VARIANT,\n"
        "    specs OBJECT NOT NULL,\n"
        "    CONSTRAINT orders_pk PRIMARY KEY (order_id),\n"
        "    CONSTRAINT orders_customer_id_fk FOREIGN KEY (customer_id) "
        "REFERENCES mini_shop_edge.customers (customer_id)\n"
        ")"
    )


def test_databricks_table_without_primary_key() -> None:
    statements = create_table_statements(SCHEMA, WarehouseName.DATABRICKS, "mini_shop")

    assert statements[2] == (
        "CREATE OR REPLACE TABLE mini_shop.events (\n"
        "    customer_id BIGINT NOT NULL,\n"
        "    event_ts TIMESTAMP NOT NULL,\n"
        "    event_name STRING NOT NULL,\n"
        "    payload VARIANT,\n"
        "    CONSTRAINT events_customer_id_fk FOREIGN KEY (customer_id) "
        "REFERENCES mini_shop.customers (customer_id)\n"
        ")"
    )


def test_parents_are_created_before_children() -> None:
    statements = create_table_statements(SCHEMA, WarehouseName.DATABRICKS, "mini_shop")

    assert statements[0].startswith("CREATE OR REPLACE TABLE mini_shop.customers ")


@pytest.mark.parametrize("dialect", [WarehouseName.SNOWFLAKE, WarehouseName.DATABRICKS])
def test_statements_parse_in_their_dialect(dialect: WarehouseName) -> None:
    for statement in create_table_statements(SCHEMA, dialect, "mini_shop"):
        sqlglot.parse_one(statement, read=dialect.value)


def test_duckdb_has_no_emitter() -> None:
    with pytest.raises(ValueError, match="no DDL emitter"):
        create_table_statements(SCHEMA, WarehouseName.DUCKDB, "mini_shop")


def test_self_reference_is_added_after_the_table_exists() -> None:
    org = load_schema(Path(__file__).parent / "fixtures" / "mini_org.yaml")
    statements = create_table_statements(org, WarehouseName.DATABRICKS, "mini_org")

    create_index = statements.index(
        next(
            sql
            for sql in statements
            if sql.startswith("CREATE OR REPLACE TABLE mini_org.employees")
        )
    )
    assert "manager_id_fk" not in statements[create_index]
    assert "departments (department_id)" in statements[create_index]
    assert statements[create_index + 1] == (
        "ALTER TABLE mini_org.employees ADD CONSTRAINT employees_manager_id_fk "
        "FOREIGN KEY (manager_id) REFERENCES mini_org.employees (employee_id)"
    )
    alters = [sql for sql in statements if sql.startswith("ALTER")]
    assert len(alters) == 1
