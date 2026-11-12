"""Sampling failures for hand labels, labels.csv, and agreement with auto labels."""

from dataclasses import replace
from pathlib import Path

import pytest
from question_files import question_data
from test_harness_store import ROW

from sqlbench.analysis.hand_labels import (
    HandLabel,
    RootCause,
    agreement,
    labels_for_run,
    read_labels,
    sample_failures,
    task_key,
    write_label_template,
    write_sheet,
)
from sqlbench.harness.store import ResultRow
from sqlbench.questions.model import Question


def failure(model_id: str, warehouse: str, number: int) -> ResultRow:
    return replace(
        ROW,
        model_id=model_id,
        warehouse=warehouse,
        question_id=f"mini_shop_{number:03d}",
        outcome="wrong_result",
    )


def test_sampling_takes_from_every_stratum_before_going_deeper() -> None:
    rows = [failure("big", "snowflake", n) for n in range(1, 21)]
    rows += [failure("small", "snowflake", 1), failure("small", "databricks", 2)]
    rows.append(replace(failure("big", "databricks", 3), outcome="correct"))

    picked = sample_failures(rows, 5)

    models = sorted((row.model_id, row.warehouse) for row in picked)
    assert ("small", "snowflake") in models
    assert ("small", "databricks") in models
    assert len(picked) == 5
    assert all(row.outcome != "correct" for row in picked)
    assert picked == sample_failures(list(reversed(rows)), 5)


def test_sampling_stops_when_failures_run_out() -> None:
    rows = [failure("a", "snowflake", 1), failure("b", "snowflake", 2)]

    assert len(sample_failures(rows, 80)) == 2


def test_sheet_and_template_list_the_same_failures(tmp_path: Path) -> None:
    rows = [failure("a", "snowflake", 1), failure("b", "databricks", 1)]
    question = Question.model_validate(question_data())

    write_sheet(rows, {"mini_shop_001": question}, tmp_path / "sheet.md")
    write_label_template(rows, tmp_path / "labels.csv")

    sheet = (tmp_path / "sheet.md").read_text()
    assert "## 01. mini_shop_001 on snowflake (a)" in sheet
    assert "GROUP BY segment" in sheet
    assert read_labels(tmp_path / "labels.csv") == []


def test_labels_are_read_and_checked(tmp_path: Path) -> None:
    path = tmp_path / "labels.csv"
    path.write_text(
        "run_id,model_id,warehouse,question_id,root_cause,note\n"
        "r,a,snowflake,q1,wrong_join, joined twice \n"
        "r,a,snowflake,q2,,\n"
    )

    assert read_labels(path) == [
        HandLabel("r", ("a", "snowflake", "q1"), RootCause.WRONG_JOIN, "joined twice")
    ]

    path.write_text("run_id,model_id,warehouse,question_id,root_cause,note\nr,a,s,q,bad_cause,\n")
    with pytest.raises(ValueError, match="labels.csv:2: unknown root cause 'bad_cause'"):
        read_labels(path)


def test_agreement_counts_only_auto_labels_with_a_matching_cause() -> None:
    auto = {
        ("a", "snowflake", "q1"): ("dialect_leak",),
        ("a", "snowflake", "q2"): ("dialect_leak", "edge:row_subset"),
        ("a", "snowflake", "q3"): ("semantics_trap",),
        ("a", "snowflake", "q4"): ("lucky_match",),
    }
    hand = [
        HandLabel("r", ("a", "snowflake", "q1"), RootCause.DIALECT_SYNTAX, ""),
        HandLabel("r", ("a", "snowflake", "q2"), RootCause.WRONG_JOIN, ""),
        HandLabel("r", ("a", "snowflake", "q3"), RootCause.SEMANTIC_TRAP, ""),
        HandLabel("r", ("a", "snowflake", "q4"), RootCause.TIE_HANDLING, ""),
    ]

    results = {result.auto_label: result for result in agreement(auto, hand)}

    assert set(results) == {"dialect_leak", "semantics_trap"}
    assert (results["dialect_leak"].labelled, results["dialect_leak"].agreeing) == (2, 1)
    assert (results["semantics_trap"].labelled, results["semantics_trap"].agreeing) == (1, 1)


def test_a_label_carries_over_only_for_the_same_failing_sql() -> None:
    old = [failure("a", "snowflake", 1), failure("a", "snowflake", 2), failure("a", "snowflake", 3)]
    labels = [HandLabel("old", task_key(row), RootCause.WRONG_JOIN, "") for row in old]
    new = [
        old[0],
        replace(old[1], predicted_sql="SELECT 2"),
        replace(old[2], outcome="correct"),
    ]

    kept = labels_for_run(labels, {task_key(row): row for row in old}, new)

    assert [label.key for label in kept] == [task_key(old[0])]
