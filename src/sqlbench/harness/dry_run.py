"""Dry-run stand-ins: the local DuckDB copy for a warehouse, a model that answers with gold.

A dry run exercises the whole pipeline (prompts, cache, extraction, guard,
execution, scoring, storage, resume) without spending anything. Warehouse
SQL is transpiled to DuckDB with sqlglot; a gold that doesn't transpile or
run there is reported as a gold failure for that task.
"""

import sqlglot
from sqlglot.errors import SqlglotError

from sqlbench.models.clients import Completion
from sqlbench.models.config import ModelConfig, Provider
from sqlbench.warehouses.base import (
    ErrorCategory,
    QueryError,
    QueryResult,
    Warehouse,
    WarehouseName,
)

DRY_RUN_MODEL = ModelConfig(
    id="dry-run",
    provider=Provider.OLLAMA,
    model="gold-answering-fake",
    version="dry-run",
    max_tokens=1,
    temperature=0,
    input_usd_per_million=0.0,
    output_usd_per_million=0.0,
    requests_per_minute=600000,
)


class DryRunWarehouse:
    """Runs one warehouse dialect's SQL on DuckDB by transpiling it first."""

    def __init__(self, inner: Warehouse, dialect: WarehouseName) -> None:
        self.name = dialect
        self._inner = inner
        self._dialect = dialect

    def execute(self, sql: str, timeout_s: int) -> QueryResult:
        try:
            statements = sqlglot.transpile(sql, read=self._dialect.value, write="duckdb")
        except SqlglotError as error:
            message = f"dry run: can't transpile to DuckDB: {error}"
            return QueryResult(elapsed_ms=0.0, error=QueryError(ErrorCategory.SYNTAX, message))
        # Several statements stay several, so the read-only guard still rejects them.
        return self._inner.execute(";\n".join(statements), timeout_s)

    def close(self) -> None:
        self._inner.close()


class GoldAnsweringModel:
    """Answers each prompt with the gold SQL of the question it was rendered for."""

    def __init__(self, answers: dict[str, str]) -> None:
        self._answers = answers

    def complete(self, prompt: str) -> Completion:
        sql = self._answers[prompt]
        return Completion(f"```sql\n{sql}\n```", len(prompt) // 4, 10, "dry-run", "stop")
