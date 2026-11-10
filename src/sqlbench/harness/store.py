"""results/results.duckdb: runs, one row per scored prediction, and cached gold results.

Every result row repeats the run's fingerprint (git sha, prompt, question
set, data and scorer version), so a row can be read on its own without
joining back to the run.
"""

import threading
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import duckdb

from sqlbench.harness.serialize import result_from_json, result_to_json
from sqlbench.warehouses.base import QueryResult

SCHEMA = [
    """
    CREATE TABLE IF NOT EXISTS runs (
        run_id VARCHAR PRIMARY KEY,
        git_sha VARCHAR NOT NULL,
        prompt_sha256 VARCHAR NOT NULL,
        question_set_sha256 VARCHAR NOT NULL,
        data_manifest_sha256 VARCHAR NOT NULL,
        scorer_version INTEGER NOT NULL,
        models VARCHAR[] NOT NULL,
        warehouses VARCHAR[] NOT NULL,
        created_at TIMESTAMP NOT NULL DEFAULT current_timestamp
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS results (
        run_id VARCHAR NOT NULL,
        git_sha VARCHAR NOT NULL,
        prompt_sha256 VARCHAR NOT NULL,
        question_set_sha256 VARCHAR NOT NULL,
        data_manifest_sha256 VARCHAR NOT NULL,
        scorer_version INTEGER NOT NULL,
        model_id VARCHAR NOT NULL,
        model_version VARCHAR NOT NULL,
        served_model VARCHAR,
        warehouse VARCHAR NOT NULL,
        question_id VARCHAR NOT NULL,
        category VARCHAR NOT NULL,
        divergence VARCHAR NOT NULL,
        features VARCHAR[] NOT NULL,
        outcome_main VARCHAR NOT NULL,
        outcome_edge VARCHAR NOT NULL,
        outcome VARCHAR NOT NULL,
        lucky_match BOOLEAN NOT NULL,
        near_miss VARCHAR[] NOT NULL,
        predicted_sql VARCHAR,
        error_category VARCHAR,
        error_message VARCHAR,
        latency_main_ms DOUBLE,
        latency_edge_ms DOUBLE,
        input_tokens BIGINT NOT NULL,
        output_tokens BIGINT NOT NULL,
        cost_usd DOUBLE NOT NULL,
        response_cached BOOLEAN NOT NULL,
        created_at TIMESTAMP NOT NULL DEFAULT current_timestamp,
        PRIMARY KEY (run_id, model_id, warehouse, question_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS gold_results (
        warehouse VARCHAR NOT NULL,
        profile VARCHAR NOT NULL,
        question_id VARCHAR NOT NULL,
        gold_sql_sha256 VARCHAR NOT NULL,
        data_sha256 VARCHAR NOT NULL,
        order_matters BOOLEAN NOT NULL,
        result_json VARCHAR NOT NULL,
        result_sha256 VARCHAR NOT NULL,
        created_at TIMESTAMP NOT NULL DEFAULT current_timestamp,
        PRIMARY KEY (warehouse, profile, question_id, gold_sql_sha256, data_sha256)
    )
    """,
]


@dataclass(frozen=True)
class RunInfo:
    """What a run's results depend on. Resuming requires all of it to match."""

    run_id: str
    git_sha: str
    prompt_sha256: str
    question_set_sha256: str
    data_manifest_sha256: str
    scorer_version: int
    models: tuple[str, ...]
    warehouses: tuple[str, ...]

    def fingerprint(self) -> tuple[str, str, str, str, int]:
        return (
            self.git_sha,
            self.prompt_sha256,
            self.question_set_sha256,
            self.data_manifest_sha256,
            self.scorer_version,
        )


@dataclass(frozen=True)
class ResultRow:
    run_id: str
    git_sha: str
    prompt_sha256: str
    question_set_sha256: str
    data_manifest_sha256: str
    scorer_version: int
    model_id: str
    model_version: str
    served_model: str | None
    warehouse: str
    question_id: str
    category: str
    divergence: str
    features: tuple[str, ...]
    outcome_main: str
    outcome_edge: str
    outcome: str
    lucky_match: bool
    # Near-miss labels, prefixed with the dataset: "main:row_subset".
    near_miss: tuple[str, ...]
    predicted_sql: str | None
    error_category: str | None
    error_message: str | None
    latency_main_ms: float | None
    latency_edge_ms: float | None
    input_tokens: int
    output_tokens: int
    cost_usd: float
    response_cached: bool


@dataclass(frozen=True)
class GoldKey:
    warehouse: str
    profile: str
    question_id: str
    gold_sql_sha256: str
    # The dataset's manifest hash: reloaded data with different content is a new key.
    data_sha256: str


@dataclass(frozen=True)
class StoredGold:
    key: GoldKey
    order_matters: bool
    result: QueryResult
    result_sha256: str


RESULT_COLUMNS = [field.name for field in fields(ResultRow)]


class ResultStore:
    """Thread-safe: worker threads score in parallel and write through one lock."""

    def __init__(self, path: Path | str) -> None:
        if isinstance(path, Path):
            path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = duckdb.connect(str(path))
        self._lock = threading.Lock()
        for statement in SCHEMA:
            self._connection.execute(statement)

    def create_run(self, run: RunInfo) -> None:
        with self._lock:
            self._connection.execute(
                "INSERT INTO runs (run_id, git_sha, prompt_sha256, question_set_sha256, "
                "data_manifest_sha256, scorer_version, models, warehouses) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    run.run_id,
                    run.git_sha,
                    run.prompt_sha256,
                    run.question_set_sha256,
                    run.data_manifest_sha256,
                    run.scorer_version,
                    list(run.models),
                    list(run.warehouses),
                ],
            )

    def get_run(self, run_id: str) -> RunInfo | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT run_id, git_sha, prompt_sha256, question_set_sha256, "
                "data_manifest_sha256, scorer_version, models, warehouses "
                "FROM runs WHERE run_id = ?",
                [run_id],
            ).fetchone()
        if row is None:
            return None
        run, git_sha, prompt_sha, question_sha, data_sha, scorer, models, warehouses = row
        return RunInfo(
            run,
            git_sha,
            prompt_sha,
            question_sha,
            data_sha,
            scorer,
            tuple(models),
            tuple(warehouses),
        )

    def add_result(self, result: ResultRow) -> None:
        values: list[Any] = []
        for name in RESULT_COLUMNS:
            value = getattr(result, name)
            values.append(list(value) if isinstance(value, tuple) else value)
        placeholders = ", ".join("?" for _ in RESULT_COLUMNS)
        with self._lock:
            self._connection.execute(
                f"INSERT INTO results ({', '.join(RESULT_COLUMNS)}) VALUES ({placeholders})",
                values,
            )

    def completed(self, run_id: str) -> set[tuple[str, str, str]]:
        """(model_id, warehouse, question_id) already stored for the run."""
        with self._lock:
            rows = self._connection.execute(
                "SELECT model_id, warehouse, question_id FROM results WHERE run_id = ?",
                [run_id],
            ).fetchall()
        return {(model_id, warehouse, question_id) for model_id, warehouse, question_id in rows}

    def results(self, run_id: str) -> list[ResultRow]:
        with self._lock:
            rows = self._connection.execute(
                f"SELECT {', '.join(RESULT_COLUMNS)} FROM results WHERE run_id = ? "
                "ORDER BY model_id, warehouse, question_id",
                [run_id],
            ).fetchall()
        stored = []
        for row in rows:
            values = dict(zip(RESULT_COLUMNS, row, strict=True))
            values["features"] = tuple(values["features"])
            values["near_miss"] = tuple(values["near_miss"])
            stored.append(ResultRow(**values))
        return stored

    def get_gold(self, key: GoldKey) -> StoredGold | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT order_matters, result_json, result_sha256 FROM gold_results "
                "WHERE warehouse = ? AND profile = ? AND question_id = ? "
                "AND gold_sql_sha256 = ? AND data_sha256 = ?",
                [key.warehouse, key.profile, key.question_id, key.gold_sql_sha256, key.data_sha256],
            ).fetchone()
        if row is None:
            return None
        order_matters, result_json, result_sha = row
        return StoredGold(key, order_matters, result_from_json(result_json), result_sha)

    def put_gold(self, gold: StoredGold) -> None:
        key = gold.key
        with self._lock:
            self._connection.execute(
                "INSERT INTO gold_results (warehouse, profile, question_id, gold_sql_sha256, "
                "data_sha256, order_matters, result_json, result_sha256) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT DO NOTHING",
                [
                    key.warehouse,
                    key.profile,
                    key.question_id,
                    key.gold_sql_sha256,
                    key.data_sha256,
                    gold.order_matters,
                    result_to_json(gold.result),
                    gold.result_sha256,
                ],
            )

    def all_gold(self) -> list[StoredGold]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT warehouse, profile, question_id, gold_sql_sha256, data_sha256, "
                "order_matters, result_json, result_sha256 FROM gold_results "
                "ORDER BY warehouse, profile, question_id"
            ).fetchall()
        stored = []
        for row in rows:
            key = GoldKey(*row[:5])
            stored.append(StoredGold(key, row[5], result_from_json(row[6]), row[7]))
        return stored

    def close(self) -> None:
        self._connection.close()
