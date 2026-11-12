"""Hand labels: a stratified sample of failures, a sheet to read them, results/labels.csv.

The sample takes failures from every (model, warehouse) in turn, so a
model with many failures can't crowd out the others. Within a stratum the
order is a hash of the task, so the sample is the same on every run.
"""

import csv
import hashlib
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from sqlbench.harness.store import ResultRow
from sqlbench.questions.model import Question
from sqlbench.warehouses.base import WarehouseName

CSV_FIELDS = ["run_id", "model_id", "warehouse", "question_id", "root_cause", "note"]

TaskKey = tuple[str, str, str]


class RootCause(StrEnum):
    WRONG_JOIN = "wrong_join"
    WRONG_GRAIN = "wrong_grain"
    WRONG_FILTER = "wrong_filter"
    NULL_HANDLING = "null_handling"
    TIE_HANDLING = "tie_handling"
    DATE_BOUNDARY = "date_boundary"
    DIALECT_SYNTAX = "dialect_syntax"
    SEMANTIC_TRAP = "semantic_trap"
    OTHER = "other"


# Where an automatic label and a hand label describe the same cause.
AUTO_TO_ROOT_CAUSE = {
    "dialect_leak": RootCause.DIALECT_SYNTAX,
    "unknown_function": RootCause.DIALECT_SYNTAX,
    "semantics_trap": RootCause.SEMANTIC_TRAP,
}


@dataclass(frozen=True)
class HandLabel:
    # The run whose failure was labelled; a later run may reuse the label.
    run_id: str
    key: TaskKey
    root_cause: RootCause
    note: str


@dataclass(frozen=True)
class Agreement:
    auto_label: str
    root_cause: RootCause
    labelled: int
    agreeing: int


def task_key(row: ResultRow) -> TaskKey:
    return (row.model_id, row.warehouse, row.question_id)


def order_key(row: ResultRow) -> str:
    return hashlib.sha256("/".join(task_key(row)).encode()).hexdigest()


def sample_failures(rows: list[ResultRow], size: int) -> list[ResultRow]:
    """Up to size failures, one from each (model, warehouse) in turn."""
    strata: dict[tuple[str, str], list[ResultRow]] = {}
    for row in rows:
        if row.outcome == "correct":
            continue
        strata.setdefault((row.model_id, row.warehouse), []).append(row)
    for members in strata.values():
        members.sort(key=order_key)

    picked: list[ResultRow] = []
    depth = 0
    while len(picked) < size:
        took_any = False
        for stratum in sorted(strata):
            members = strata[stratum]
            if depth < len(members) and len(picked) < size:
                picked.append(members[depth])
                took_any = True
        if not took_any:
            break
        depth += 1
    picked.sort(key=order_key)
    return picked


def sheet_entry(number: int, row: ResultRow, question: Question) -> list[str]:
    warehouse = WarehouseName(row.warehouse)
    lines = [
        f"## {number:02d}. {row.question_id} on {row.warehouse} ({row.model_id})",
        "",
        question.question,
        "",
        f"outcome: {row.outcome} (main {row.outcome_main}, edge {row.outcome_edge})",
    ]
    if row.near_miss:
        lines.append(f"near miss: {', '.join(row.near_miss)}")
    if row.error_message:
        lines.append(f"error: {row.error_message.splitlines()[0][:300]}")
    lines += [
        "",
        "Gold SQL:",
        "```sql",
        question.gold_sql.for_warehouse(warehouse).strip(),
        "```",
        "Predicted SQL:",
        "```sql",
        (row.predicted_sql or "(no SQL)").strip(),
        "```",
        "",
    ]
    return lines


def write_sheet(sample: list[ResultRow], questions: dict[str, Question], path: Path) -> None:
    causes = ", ".join(cause.value for cause in RootCause)
    lines = [
        f"# Failure root causes ({len(sample)} failures)",
        "",
        f"Label each in labels.csv with one of: {causes}.",
        "",
    ]
    for number, row in enumerate(sample, start=1):
        lines += sheet_entry(number, row, questions[row.question_id])
    path.write_text("\n".join(lines))


def write_label_template(sample: list[ResultRow], path: Path) -> None:
    """labels.csv with an empty root_cause per sampled failure."""
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in sample:
            model_id, warehouse, question_id = task_key(row)
            writer.writerow(
                {
                    "run_id": row.run_id,
                    "model_id": model_id,
                    "warehouse": warehouse,
                    "question_id": question_id,
                    "root_cause": "",
                    "note": "",
                }
            )


def read_labels(path: Path) -> list[HandLabel]:
    """Filled-in rows of labels.csv; raises ValueError on an unknown root cause."""
    labels = []
    with path.open(newline="") as handle:
        for line_number, record in enumerate(csv.DictReader(handle), start=2):
            cause = record["root_cause"].strip()
            if not cause:
                continue
            try:
                root_cause = RootCause(cause)
            except ValueError as error:
                raise ValueError(f"{path}:{line_number}: unknown root cause {cause!r}") from error
            key = (record["model_id"], record["warehouse"], record["question_id"])
            labels.append(HandLabel(record["run_id"], key, root_cause, record["note"].strip()))
    return labels


def labels_for_run(
    labels: list[HandLabel], labelled_rows: dict[TaskKey, ResultRow], rows: list[ResultRow]
) -> list[HandLabel]:
    """The labels that still describe a failure in rows.

    labelled_rows are the rows the labels were made on. A label carries over to
    another run only if the same task failed there with the same predicted SQL.
    """
    current = {task_key(row): row for row in rows}
    kept = []
    for label in labels:
        row = current.get(label.key)
        labelled = labelled_rows.get(label.key)
        if row is None or labelled is None or row.outcome == "correct":
            continue
        if row.predicted_sql == labelled.predicted_sql:
            kept.append(label)
    return kept


def agreement(auto: dict[TaskKey, tuple[str, ...]], hand: Iterable[HandLabel]) -> list[Agreement]:
    """For each auto label with a matching root cause: how often the hand label agrees."""
    labelled: dict[str, int] = {}
    agreeing: dict[str, int] = {}
    for label in hand:
        for auto_label in auto.get(label.key, ()):
            expected = AUTO_TO_ROOT_CAUSE.get(auto_label)
            if expected is None:
                continue
            labelled[auto_label] = labelled.get(auto_label, 0) + 1
            if label.root_cause == expected:
                agreeing[auto_label] = agreeing.get(auto_label, 0) + 1
    results = []
    for auto_label in sorted(labelled):
        results.append(
            Agreement(
                auto_label,
                AUTO_TO_ROOT_CAUSE[auto_label],
                labelled[auto_label],
                agreeing.get(auto_label, 0),
            )
        )
    return results
