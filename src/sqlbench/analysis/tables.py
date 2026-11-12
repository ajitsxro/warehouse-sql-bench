"""The numbers in the report, computed from stored rows. Nothing here formats text.

Accuracy is per question: a (model, warehouse) row is 1.0 when correct,
else 0.0. Pooled over warehouses, a question counts 0.5 if a model got it
right on one warehouse only.
"""

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass

from sqlbench.analysis.labels import Label
from sqlbench.analysis.stats import (
    Interval,
    mean_interval,
    paired_difference_interval,
    unpaired_difference_interval,
)
from sqlbench.harness.store import ResultRow
from sqlbench.questions.model import Question

WAREHOUSES = ("snowflake", "databricks")
FAILURE_LABELS = (
    Label.DIALECT_LEAK,
    Label.UNKNOWN_FUNCTION,
    Label.SCHEMA_HALLUCINATION,
    Label.SEMANTICS_TRAP,
    Label.LUCKY_MATCH,
)
TaskKey = tuple[str, str, str]


@dataclass(frozen=True)
class RunData:
    rows: list[ResultRow]
    questions: dict[str, Question]
    labels: dict[TaskKey, tuple[str, ...]]

    @property
    def models(self) -> list[str]:
        return sorted({row.model_id for row in self.rows})

    @property
    def question_ids(self) -> list[str]:
        return sorted({row.question_id for row in self.rows})

    def row(self, model_id: str, warehouse: str, question_id: str) -> ResultRow:
        for row in self.rows:
            if (row.model_id, row.warehouse, row.question_id) == (model_id, warehouse, question_id):
                return row
        raise KeyError((model_id, warehouse, question_id))

    def labels_of(self, row: ResultRow) -> tuple[str, ...]:
        return self.labels.get((row.model_id, row.warehouse, row.question_id), ())


def is_correct(row: ResultRow) -> float:
    return 1.0 if row.outcome == "correct" else 0.0


def scores(data: RunData, model_id: str, warehouse: str, question_ids: list[str]) -> list[float]:
    by_question = {}
    for row in data.rows:
        if row.model_id == model_id and row.warehouse == warehouse:
            by_question[row.question_id] = is_correct(row)
    return [by_question[question_id] for question_id in question_ids]


def pooled_scores(data: RunData, model_id: str, question_ids: list[str]) -> list[float]:
    """Per question, the share of the two warehouses the model got right."""
    first = scores(data, model_id, WAREHOUSES[0], question_ids)
    second = scores(data, model_id, WAREHOUSES[1], question_ids)
    return [(a + b) / 2 for a, b in zip(first, second, strict=True)]


def questions_where(data: RunData, keep: Callable[[Question], bool]) -> list[str]:
    return [question_id for question_id in data.question_ids if keep(data.questions[question_id])]


@dataclass(frozen=True)
class WarehouseAccuracy:
    model_id: str
    by_warehouse: dict[str, Interval]
    # Snowflake minus Databricks, paired over the same questions.
    gap: Interval


def accuracy_by_warehouse(data: RunData) -> list[WarehouseAccuracy]:
    """Sorted best first by pooled accuracy."""
    question_ids = data.question_ids
    results = []
    for model_id in data.models:
        per_warehouse = {}
        for warehouse in WAREHOUSES:
            per_warehouse[warehouse] = mean_interval(
                scores(data, model_id, warehouse, question_ids)
            )
        gap = paired_difference_interval(
            scores(data, model_id, WAREHOUSES[0], question_ids),
            scores(data, model_id, WAREHOUSES[1], question_ids),
        )
        results.append(WarehouseAccuracy(model_id, per_warehouse, gap))
    results.sort(key=lambda entry: -sum(i.estimate for i in entry.by_warehouse.values()))
    return results


@dataclass(frozen=True)
class NeighbourGap:
    warehouse: str
    better: str
    worse: str
    gap: Interval


def neighbour_gaps(data: RunData, ranked: list[WarehouseAccuracy]) -> list[NeighbourGap]:
    """Each model against the next one down, per warehouse, paired over questions."""
    question_ids = data.question_ids
    gaps = []
    for warehouse in WAREHOUSES:
        order = sorted(ranked, key=lambda entry: -entry.by_warehouse[warehouse].estimate)
        for better, worse in zip(order, order[1:], strict=False):
            gap = paired_difference_interval(
                scores(data, better.model_id, warehouse, question_ids),
                scores(data, worse.model_id, warehouse, question_ids),
            )
            gaps.append(NeighbourGap(warehouse, better.model_id, worse.model_id, gap))
    return gaps


@dataclass(frozen=True)
class GroupGap:
    model_id: str
    first: Interval
    second: Interval
    # first minus second, resampling each group's questions separately.
    gap: Interval


def group_gaps(
    data: RunData, first: Callable[[Question], bool], second: Callable[[Question], bool]
) -> list[GroupGap]:
    first_ids = questions_where(data, first)
    second_ids = questions_where(data, second)
    results = []
    for model_id in data.models:
        first_scores = pooled_scores(data, model_id, first_ids)
        second_scores = pooled_scores(data, model_id, second_ids)
        results.append(
            GroupGap(
                model_id,
                mean_interval(first_scores),
                mean_interval(second_scores),
                unpaired_difference_interval(first_scores, second_scores),
            )
        )
    return results


def accuracy_by_group(
    data: RunData, groups: list[str], group_of: Callable[[Question], list[str]]
) -> dict[str, dict[str, tuple[float, int]]]:
    """model -> group -> (pooled accuracy, number of questions)."""
    table: dict[str, dict[str, tuple[float, int]]] = {}
    for model_id in data.models:
        table[model_id] = {}
        for group in groups:
            ids = [
                question_id
                for question_id in data.question_ids
                if group in group_of(data.questions[question_id])
            ]
            if not ids:
                continue
            pooled = pooled_scores(data, model_id, ids)
            table[model_id][group] = (sum(pooled) / len(pooled), len(ids))
    return table


def failure_outcomes(data: RunData) -> dict[str, Counter[str]]:
    """warehouse -> outcome -> count, failures only."""
    counts: dict[str, Counter[str]] = {warehouse: Counter() for warehouse in WAREHOUSES}
    for row in data.rows:
        if row.outcome != "correct":
            counts[row.warehouse][row.outcome] += 1
    return counts


def failure_labels(data: RunData) -> dict[str, Counter[str]]:
    """warehouse -> automatic label -> count."""
    counts: dict[str, Counter[str]] = {warehouse: Counter() for warehouse in WAREHOUSES}
    for row in data.rows:
        for label in data.labels_of(row):
            if label in FAILURE_LABELS:
                counts[row.warehouse][label] += 1
    return counts


@dataclass(frozen=True)
class DivergenceFailures:
    divergence: str
    warehouse: str
    tasks: int
    failures: int
    outcomes: Counter[str]


def failures_by_divergence(data: RunData) -> list[DivergenceFailures]:
    results = []
    for divergence in ("none", "syntax", "semantics"):
        for warehouse in WAREHOUSES:
            rows = [
                row
                for row in data.rows
                if row.warehouse == warehouse
                and data.questions[row.question_id].divergence == divergence
            ]
            outcomes = Counter(row.outcome for row in rows if row.outcome != "correct")
            failed = sum(outcomes.values())
            results.append(DivergenceFailures(divergence, warehouse, len(rows), failed, outcomes))
    return results


@dataclass(frozen=True)
class SplitQuestion:
    question_id: str
    right_on: str
    failing: ResultRow


def split_questions(data: RunData) -> dict[str, list[SplitQuestion]]:
    """model -> questions right on exactly one warehouse."""
    results: dict[str, list[SplitQuestion]] = {}
    for model_id in data.models:
        results[model_id] = []
        for question_id in data.question_ids:
            first = data.row(model_id, WAREHOUSES[0], question_id)
            second = data.row(model_id, WAREHOUSES[1], question_id)
            if is_correct(first) == is_correct(second):
                continue
            if is_correct(first):
                results[model_id].append(SplitQuestion(question_id, WAREHOUSES[0], second))
            else:
                results[model_id].append(SplitQuestion(question_id, WAREHOUSES[1], first))
    return results


@dataclass(frozen=True)
class LuckyRate:
    model_id: str
    lucky: int
    tasks: int
    main_only_accuracy: float
    accuracy: float

    @property
    def overstatement(self) -> float:
        return self.main_only_accuracy - self.accuracy


def lucky_rates(data: RunData) -> list[LuckyRate]:
    results = []
    for model_id in data.models:
        rows = [row for row in data.rows if row.model_id == model_id]
        lucky = sum(1 for row in rows if row.lucky_match)
        main_correct = sum(1 for row in rows if row.outcome_main == "correct")
        correct = sum(1 for row in rows if row.outcome == "correct")
        results.append(
            LuckyRate(model_id, lucky, len(rows), main_correct / len(rows), correct / len(rows))
        )
    return results


# The order examples are picked in: each category takes the first matching
# failure (by a hash of the task) from a model not shown yet, if there is one.
EXAMPLE_CATEGORIES: list[tuple[str, Callable[[ResultRow, tuple[str, ...]], bool]]] = [
    ("Dialect leak", lambda row, labels: Label.DIALECT_LEAK in labels),
    ("Function from neither warehouse", lambda row, labels: Label.UNKNOWN_FUNCTION in labels),
    ("Invented table or column", lambda row, labels: Label.SCHEMA_HALLUCINATION in labels),
    ("Semantics trap", lambda row, labels: Label.SEMANTICS_TRAP in labels),
    ("Lucky match (right on main, wrong on edge)", lambda row, labels: row.lucky_match),
    (
        "Missing or extra rows",
        lambda row, labels: any("row_subset" in t or "row_superset" in t for t in row.near_miss),
    ),
    ("Extra columns", lambda row, labels: row.outcome == "wrong_columns"),
    ("No SQL in the answer", lambda row, labels: row.outcome == "no_sql"),
]


def pick_examples(data: RunData, order: Callable[[ResultRow], str]) -> list[tuple[str, ResultRow]]:
    failures = sorted((row for row in data.rows if row.outcome != "correct"), key=order)
    shown_models: set[str] = set()
    shown_tasks: set[TaskKey] = set()
    picked = []
    for title, matches in EXAMPLE_CATEGORIES:
        candidates = []
        for row in failures:
            key = (row.model_id, row.warehouse, row.question_id)
            if key not in shown_tasks and matches(row, data.labels_of(row)):
                candidates.append(row)
        if not candidates:
            continue
        fresh = [row for row in candidates if row.model_id not in shown_models]
        chosen = (fresh or candidates)[0]
        shown_models.add(chosen.model_id)
        shown_tasks.add((chosen.model_id, chosen.warehouse, chosen.question_id))
        picked.append((title, chosen))
    return picked
