"""Report tables on a small hand-built run."""

from dataclasses import replace

from question_files import question_data
from test_harness_store import ROW

from sqlbench.analysis.hand_labels import order_key
from sqlbench.analysis.tables import (
    RunData,
    accuracy_by_warehouse,
    failures_by_divergence,
    lucky_rates,
    pick_examples,
    split_questions,
)
from sqlbench.harness.store import ResultRow
from sqlbench.questions.model import Question

QUESTIONS = {
    "mini_shop_001": Question.model_validate(question_data(id="mini_shop_001")),
    "mini_shop_002": Question.model_validate(
        question_data(
            id="mini_shop_002",
            category="dialect_sensitive",
            divergence="syntax",
            divergence_note="FLATTEN vs explode.",
            difficulty="hard",
        )
    ),
    "mini_shop_003": Question.model_validate(
        question_data(
            id="mini_shop_003",
            category="dialect_sensitive",
            divergence="semantics",
            divergence_note="NULL order (probe null_order_desc).",
        )
    ),
}
MANIFEST: dict[str, object] = {
    "run_id": "test",
    "started_at_utc": "2026-11-11T00:00:00",
    "finished_at_utc": "2026-11-11T01:00:00",
    "git_sha": "abc",
    "question_set_sha256": "q" * 64,
    "data_manifest_sha256": "d" * 64,
    "scorer_version": 1,
    "parquet_sha256": "p" * 64,
}


def row(model_id: str, warehouse: str, question_id: str, outcome: str) -> ResultRow:
    return replace(
        ROW,
        model_id=model_id,
        warehouse=warehouse,
        question_id=question_id,
        outcome=outcome,
        outcome_main=outcome,
        outcome_edge=outcome,
        lucky_match=False,
        near_miss=(),
        error_message=None,
    )


def run_data() -> RunData:
    rows = []
    for warehouse in ("snowflake", "databricks"):
        for question_id in QUESTIONS:
            rows.append(row("strong", warehouse, question_id, "correct"))
    rows += [
        row("weak", "snowflake", "mini_shop_001", "correct"),
        row("weak", "databricks", "mini_shop_001", "correct"),
        replace(
            row("weak", "snowflake", "mini_shop_002", "wrong_result"),
            outcome_main="correct",
            lucky_match=True,
            near_miss=("edge:row_subset",),
        ),
        replace(
            row("weak", "databricks", "mini_shop_002", "syntax_error"),
            error_message="Could not resolve FLATTEN to a table-valued function.",
        ),
        row("weak", "snowflake", "mini_shop_003", "correct"),
        row("weak", "databricks", "mini_shop_003", "wrong_result"),
    ]
    labels = {
        ("weak", "databricks", "mini_shop_002"): ("dialect_leak",),
        ("weak", "snowflake", "mini_shop_002"): ("lucky_match", "edge:row_subset"),
    }
    return RunData(rows, QUESTIONS, labels)


def test_accuracy_ranks_models_and_pairs_the_warehouses() -> None:
    ranked = accuracy_by_warehouse(run_data())

    assert [entry.model_id for entry in ranked] == ["strong", "weak"]
    weak = ranked[1]
    assert weak.by_warehouse["snowflake"].estimate == 2 / 3
    assert weak.by_warehouse["databricks"].estimate == 1 / 3
    assert weak.gap.estimate == 1 / 3


def test_split_questions_name_the_failing_side() -> None:
    splits = split_questions(run_data())

    assert splits["strong"] == []
    assert [(s.question_id, s.right_on) for s in splits["weak"]] == [("mini_shop_003", "snowflake")]
    assert splits["weak"][0].failing.warehouse == "databricks"


def test_lucky_rate_is_what_main_alone_would_have_added() -> None:
    weak = [rate for rate in lucky_rates(run_data()) if rate.model_id == "weak"][0]

    assert weak.lucky == 1
    assert weak.main_only_accuracy == 4 / 6
    assert weak.accuracy == 3 / 6
    assert abs(weak.overstatement - 1 / 6) < 1e-9


def test_failures_are_grouped_by_divergence() -> None:
    by_key = {(f.divergence, f.warehouse): f for f in failures_by_divergence(run_data())}

    assert by_key[("syntax", "databricks")].failures == 1
    assert by_key[("syntax", "databricks")].outcomes == {"syntax_error": 1}
    assert by_key[("none", "snowflake")].failures == 0


def test_examples_cover_each_category_once() -> None:
    picked = pick_examples(run_data(), order_key)

    titles = [title for title, _ in picked]
    assert titles[0] == "Dialect leak"
    assert "Lucky match (right on main, wrong on edge)" in titles
    assert len({(r.model_id, r.warehouse, r.question_id) for _, r in picked}) == len(picked)
