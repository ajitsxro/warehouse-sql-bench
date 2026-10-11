"""Snowflake adapter using key-pair auth.

Benchmark queries log in as a read-only user whose role can only SELECT from
the benchmark database, a second safety layer behind the read-only guard.
Loading logs in as the owner user. Both users share one key pair.
"""

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

SHARED_ENV_VARS = ["SNOWFLAKE_ACCOUNT", "SNOWFLAKE_PRIVATE_KEY_PATH", "SNOWFLAKE_WAREHOUSE"]
OWNER_ENV_VARS = ["SNOWFLAKE_USER", "SNOWFLAKE_ROLE"]
READER_ENV_VARS = ["SNOWFLAKE_READER_USER", "SNOWFLAKE_READER_ROLE"]
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
    """Connect as the read-only user, for benchmark queries."""
    env = require_env(SHARED_ENV_VARS + READER_ENV_VARS)
    return connect(env, env["SNOWFLAKE_READER_USER"], env["SNOWFLAKE_READER_ROLE"], context)


def connect_owner_from_env(context: SessionContext) -> SnowflakeConnection:
    """Connect as the user that owns the benchmark database, for loading."""
    env = require_env(SHARED_ENV_VARS + OWNER_ENV_VARS)
    return connect(env, env["SNOWFLAKE_USER"], env["SNOWFLAKE_ROLE"], context)


def connect(
    env: dict[str, str], user: str, role: str, context: SessionContext
) -> SnowflakeConnection:
    return snowflake.connector.connect(
        account=env["SNOWFLAKE_ACCOUNT"],
        user=user,
        private_key_file=env["SNOWFLAKE_PRIVATE_KEY_PATH"],
        warehouse=env["SNOWFLAKE_WAREHOUSE"],
        role=role,
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
