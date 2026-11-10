"""One task end to end with scripted warehouses and a fake model."""

from typing import Any

import pytest
from question_files import question_data

from sqlbench.datagen.tables import Profile
from sqlbench.harness.executor import Task, ensure_gold, execute_task
from sqlbench.harness.store import ResultStore, RunInfo, StoredGold
from sqlbench.models.cache import ResponseCache
from sqlbench.models.clients import Completion
from sqlbench.models.config import ModelConfig, Provider
from sqlbench.models.prompt import PromptTemplate
from sqlbench.questions.model import Question
from sqlbench.scoring.outcomes import GoldError
from sqlbench.warehouses.base import (
    Column,
    ErrorCategory,
    QueryError,
    QueryResult,
    WarehouseName,
)

GOLD_SQL = "SELECT gold"
TEMPLATE = PromptTemplate("{question}", "t" * 64)
RUN = RunInfo("run-1", "abc", "p" * 64, "q" * 64, "d" * 64, 1, ("fake-1",), ("snowflake",))
MODEL = ModelConfig(
    id="fake-1",
    provider=Provider.OPENAI,
    model="fake",
    version="fake-v1",
    max_tokens=100,
    temperature=0,
    input_usd_per_million=1.0,
    output_usd_per_million=1.0,
    requests_per_minute=6000,
)

Rows = list[tuple[Any, ...]]


class ScriptedWarehouse:
    """Answers each SQL text with fixed rows or an error, and counts queries."""

    def __init__(self, answers: dict[str, Rows | QueryError]) -> None:
        self.name = WarehouseName.SNOWFLAKE
        self.answers = answers
        self.queries: list[str] = []

    def execute(self, sql: str, timeout_s: int) -> QueryResult:
        self.queries.append(sql)
        answer = self.answers[sql]
        if isinstance(answer, QueryError):
            return QueryResult(elapsed_ms=3.0, error=answer)
        width = len(answer[0]) if answer else 1
        columns = [Column(f"c{index}", "FIXED") for index in range(width)]
        return QueryResult(elapsed_ms=7.0, columns=columns, rows=answer)

    def close(self) -> None:
        return None


class FakeModel:
    def __init__(self, text: str) -> None:
        self.text = text
        self.calls = 0

    def complete(self, prompt: str) -> Completion:
        self.calls += 1
        return Completion(self.text, 1000, 20, "fake-served", "stop")


def question(**changes: Any) -> Question:
    golds = {"snowflake": GOLD_SQL, "databricks": GOLD_SQL}
    return Question.model_validate(question_data(gold_sql=golds, **changes))


def run_task(
    answer_text: str,
    main: dict[str, Rows | QueryError],
    edge: dict[str, Rows | QueryError],
    **question_changes: Any,
) -> tuple[Any, dict[Profile, ScriptedWarehouse]]:
    adapters = {Profile.MAIN: ScriptedWarehouse(main), Profile.EDGE: ScriptedWarehouse(edge)}
    store = ResultStore(":memory:")
    asked = question(**question_changes)
    golds: dict[Profile, StoredGold] = {}
    for profile, adapter in adapters.items():
        golds[profile] = ensure_gold(store, asked, WarehouseName.SNOWFLAKE, profile, adapter, "d")
    task = Task(MODEL, WarehouseName.SNOWFLAKE, asked, f"Q: {asked.question}")
    row = execute_task(
        task, RUN, FakeModel(answer_text), ResponseCache(":memory:"), TEMPLATE, golds, adapters
    )
    return row, adapters


def answer(sql: str) -> str:
    return f"Here:\n```sql\n{sql};\n```"


def test_a_correct_prediction() -> None:
    row, adapters = run_task(
        answer("SELECT p"),
        {GOLD_SQL: [(1,), (2,)], "SELECT p": [(2,), (1,)]},
        {GOLD_SQL: [(3,)], "SELECT p": [(3,)]},
    )

    assert (row.outcome_main, row.outcome_edge, row.outcome) == ("correct", "correct", "correct")
    assert row.lucky_match is False
    assert row.predicted_sql == "SELECT p"
    assert row.near_miss == ()
    assert (row.latency_main_ms, row.latency_edge_ms) == (7.0, 7.0)
    assert (row.input_tokens, row.output_tokens) == (1000, 20)
    assert row.cost_usd == pytest.approx(1020 / 1e6)
    assert row.response_cached is False
    assert row.served_model == "fake-served"
    assert adapters[Profile.MAIN].queries == [GOLD_SQL, "SELECT p"]


def test_a_lucky_prediction_passes_main_and_fails_edge() -> None:
    # Right on realistic data, wrong once edge adds a duplicate row.
    row, _ = run_task(
        answer("SELECT DISTINCT p"),
        {GOLD_SQL: [(1,), (2,)], "SELECT DISTINCT p": [(1,), (2,)]},
        {GOLD_SQL: [(1,), (1,), (2,)], "SELECT DISTINCT p": [(1,), (2,)]},
    )

    assert row.outcome_main == "correct"
    assert row.outcome_edge == "wrong_result"
    assert row.outcome == "wrong_result"
    assert row.lucky_match is True
    assert row.near_miss == ("edge:row_subset",)


def test_no_sql_runs_nothing() -> None:
    row, adapters = run_task(
        "I can't tell from the schema.", {GOLD_SQL: [(1,)]}, {GOLD_SQL: [(1,)]}
    )

    assert (row.outcome_main, row.outcome_edge, row.outcome) == ("no_sql", "no_sql", "no_sql")
    assert row.predicted_sql is None
    assert row.latency_main_ms is None
    assert adapters[Profile.MAIN].queries == [GOLD_SQL]


def test_query_errors_are_recorded() -> None:
    failure = QueryError(ErrorCategory.SYNTAX, "unknown column x")
    row, _ = run_task(
        answer("SELECT x"),
        {GOLD_SQL: [(1,)], "SELECT x": failure},
        {GOLD_SQL: [(1,)], "SELECT x": failure},
    )

    assert row.outcome == "syntax_error"
    assert (row.error_category, row.error_message) == ("syntax", "unknown column x")


def test_the_error_reported_is_from_main_first() -> None:
    row, _ = run_task(
        answer("SELECT p"),
        {GOLD_SQL: [(1,)], "SELECT p": QueryError(ErrorCategory.TIMEOUT, "slow")},
        {GOLD_SQL: [(1,)], "SELECT p": QueryError(ErrorCategory.RUNTIME, "bad cast")},
    )

    assert (row.outcome_main, row.outcome_edge) == ("timeout", "runtime_error")
    assert row.error_message == "slow"


def test_an_edge_only_error_is_recorded() -> None:
    row, _ = run_task(
        answer("SELECT p"),
        {GOLD_SQL: [(1,)], "SELECT p": [(1,)]},
        {GOLD_SQL: [(1,)], "SELECT p": QueryError(ErrorCategory.RUNTIME, "division by zero")},
    )

    assert row.lucky_match is True
    assert row.error_message == "division by zero"


def test_the_row_describes_the_question_and_run() -> None:
    row, _ = run_task(
        answer("SELECT p"),
        {GOLD_SQL: [(1,)], "SELECT p": [(1,)]},
        {GOLD_SQL: [(1,)], "SELECT p": [(1,)]},
        features=["join", "window"],
    )

    assert row.run_id == "run-1"
    assert (row.category, row.divergence) == ("portable", "none")
    assert row.features == ("join", "window")
    assert row.model_version == "fake-v1"
    assert row.scorer_version == 1


def test_gold_runs_once_and_is_then_read_from_the_store() -> None:
    store = ResultStore(":memory:")
    adapter = ScriptedWarehouse({GOLD_SQL: [(1,)]})
    asked = question()

    first = ensure_gold(store, asked, WarehouseName.SNOWFLAKE, Profile.MAIN, adapter, "d1")
    second = ensure_gold(store, asked, WarehouseName.SNOWFLAKE, Profile.MAIN, adapter, "d1")
    ensure_gold(store, asked, WarehouseName.SNOWFLAKE, Profile.MAIN, adapter, "d2")

    assert first.result_sha256 == second.result_sha256
    assert second.result.rows == [(1,)]
    assert adapter.queries == [GOLD_SQL, GOLD_SQL]


def test_failing_gold_raises() -> None:
    adapter = ScriptedWarehouse({GOLD_SQL: QueryError(ErrorCategory.SYNTAX, "boom")})

    with pytest.raises(GoldError, match="mini_shop_001 snowflake/main: syntax: boom"):
        ensure_gold(
            ResultStore(":memory:"), question(), WarehouseName.SNOWFLAKE, Profile.MAIN, adapter, "d"
        )
