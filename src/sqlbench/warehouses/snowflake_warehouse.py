"""Snowflake adapter using key-pair auth."""

import time
from collections.abc import Callable

import snowflake.connector
from snowflake.connector import SnowflakeConnection

# Public re-export the stubs don't declare; the real definition is in a private module.
from snowflake.connector.constants import FIELD_ID_TO_NAME  # type: ignore[attr-defined]
from snowflake.connector.errors import Error as SnowflakeError

from sqlbench.settings import require_env
from sqlbench.warehouses.base import (
    SESSION_TIME_ZONE,
    Column,
    ErrorCategory,
    QueryError,
    QueryResult,
    SessionContext,
    WarehouseName,
    rows_from_arrow,
)
from sqlbench.warehouses.guard import check_read_only
from sqlbench.warehouses.sqlstate import category_from_sqlstate

ENV_VARS = [
    "SNOWFLAKE_ACCOUNT",
    "SNOWFLAKE_USER",
    "SNOWFLAKE_PRIVATE_KEY_PATH",
    "SNOWFLAKE_WAREHOUSE",
    "SNOWFLAKE_ROLE",
]
QUERY_TAG = "sqlbench"
LOGIN_TIMEOUT_S = 30
# 604: "SQL execution canceled", 630: "Statement reached its statement or
# warehouse timeout". Matched by number because these are documented codes.
TIMEOUT_ERRNOS = (604, 630)


def categorize_snowflake_error(error: SnowflakeError) -> ErrorCategory:
    if error.errno in TIMEOUT_ERRNOS:
        return ErrorCategory.TIMEOUT
    return category_from_sqlstate(error.sqlstate)


def connect_from_env(context: SessionContext) -> SnowflakeConnection:
    env = require_env(ENV_VARS)
    return snowflake.connector.connect(
        account=env["SNOWFLAKE_ACCOUNT"],
        user=env["SNOWFLAKE_USER"],
        private_key_file=env["SNOWFLAKE_PRIVATE_KEY_PATH"],
        warehouse=env["SNOWFLAKE_WAREHOUSE"],
        role=env["SNOWFLAKE_ROLE"],
        database=context.database,
        schema=context.schema,
        session_parameters={"QUERY_TAG": QUERY_TAG, "TIMEZONE": SESSION_TIME_ZONE},
        login_timeout=LOGIN_TIMEOUT_S,
    )


class SnowflakeWarehouse:
    """Runs queries on Snowflake. The connection opens on the first execute."""

    name = WarehouseName.SNOWFLAKE

    def __init__(
        self,
        context: SessionContext | None = None,
        connect: Callable[[SessionContext], SnowflakeConnection] = connect_from_env,
    ) -> None:
        self._context = context or SessionContext()
        self._connect = connect
        self._connection: SnowflakeConnection | None = None
        self._session_timeout_s: int | None = None

    def execute(self, sql: str, timeout_s: int) -> QueryResult:
        rejection = check_read_only(sql, self.name)
        if rejection is not None:
            return QueryResult(elapsed_ms=0.0, error=rejection)

        start = time.perf_counter()
        try:
            if self._connection is None:
                self._connection = self._connect(self._context)
            with self._connection.cursor() as cursor:
                # The timeout is enforced server-side; only resend it when it changes.
                if self._session_timeout_s != timeout_s:
                    cursor.execute(
                        f"ALTER SESSION SET STATEMENT_TIMEOUT_IN_SECONDS = {int(timeout_s)}"
                    )
                    self._session_timeout_s = timeout_s
                # Time only the query, not connecting or session setup.
                start = time.perf_counter()
                cursor.execute(sql)
                columns = []
                for meta in cursor.description or []:
                    type_name = FIELD_ID_TO_NAME.get(meta.type_code, str(meta.type_code))
                    columns.append(Column(meta.name, type_name))
                # Microseconds keep timestamps as datetime values; the
                # nanosecond default needs pandas to represent them.
                table = cursor.fetch_arrow_all(
                    force_return_table=True, force_microsecond_precision=True
                )
                rows = rows_from_arrow(table)
        except SnowflakeError as error:
            elapsed_ms = (time.perf_counter() - start) * 1000
            category = categorize_snowflake_error(error)
            return QueryResult(elapsed_ms=elapsed_ms, error=QueryError(category, str(error)))

        elapsed_ms = (time.perf_counter() - start) * 1000
        return QueryResult(elapsed_ms=elapsed_ms, columns=columns, rows=rows)

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None
