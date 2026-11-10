"""Score a prediction against gold on one dataset, then combine main and edge."""

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from sqlbench.scoring import SCORER_VERSION
from sqlbench.scoring.diagnostics import Diagnostic, diagnose
from sqlbench.scoring.matching import MatchRules, find_column_order
from sqlbench.scoring.normalize import normalize_rows, numbers_equal
from sqlbench.warehouses.base import ErrorCategory, QueryResult


class Outcome(StrEnum):
    CORRECT = "correct"
    WRONG_RESULT = "wrong_result"
    # Gold has rows and the prediction returned none.
    EMPTY_RESULT = "empty_result"
    # More or fewer columns than gold.
    WRONG_COLUMNS = "wrong_columns"
    SYNTAX_ERROR = "syntax_error"
    RUNTIME_ERROR = "runtime_error"
    TIMEOUT = "timeout"
    # The read-only guard refused the SQL.
    BLOCKED = "blocked"
    # The model returned no SQL.
    NO_SQL = "no_sql"


# PERMISSION and OTHER come from the warehouse while the query runs.
ERROR_OUTCOMES = {
    ErrorCategory.SYNTAX: Outcome.SYNTAX_ERROR,
    ErrorCategory.RUNTIME: Outcome.RUNTIME_ERROR,
    ErrorCategory.TIMEOUT: Outcome.TIMEOUT,
    ErrorCategory.BLOCKED: Outcome.BLOCKED,
    ErrorCategory.PERMISSION: Outcome.RUNTIME_ERROR,
    ErrorCategory.OTHER: Outcome.RUNTIME_ERROR,
}


class GoldError(Exception):
    """Gold SQL failed; a question with broken gold can't score anything."""


@dataclass(frozen=True)
class DatasetScore:
    outcome: Outcome
    # For each gold column, the predicted column that matched it (when correct).
    column_order: tuple[int, ...] | None = None
    # Near-miss labels for wrong_result and wrong_columns; never affect the outcome.
    diagnostics: tuple[Diagnostic, ...] = ()


@dataclass(frozen=True)
class QuestionScore:
    main: DatasetScore
    edge: DatasetScore
    outcome: Outcome
    # Right on the realistic data, wrong on the adversarial data.
    lucky_match: bool
    scorer_version: int = SCORER_VERSION


def match_rules(order_matters: bool, tolerance: float) -> MatchRules:
    def within_tolerance(left: Any, right: Any) -> bool:
        return numbers_equal(left, right, tolerance)

    return MatchRules(order_matters, within_tolerance)


def score_dataset(
    gold: QueryResult, predicted: QueryResult | None, order_matters: bool, tolerance: float
) -> DatasetScore:
    """Outcome of one predicted result; predicted is None when the model gave no SQL.

    Checks run in this order: no SQL, error, column count, empty result,
    rows. So a prediction with the wrong columns and no rows is
    wrong_columns.
    """
    if gold.error is not None:
        raise GoldError(f"gold failed: {gold.error.category}: {gold.error.message}")
    if predicted is None:
        return DatasetScore(Outcome.NO_SQL)
    if predicted.error is not None:
        return DatasetScore(ERROR_OUTCOMES[predicted.error.category])
    gold_width = len(gold.columns)
    predicted_width = len(predicted.columns)
    gold_rows = normalize_rows(gold.rows, [column.type for column in gold.columns])
    predicted_rows = normalize_rows(predicted.rows, [column.type for column in predicted.columns])
    if predicted_width != gold_width:
        labels = diagnose(
            gold_rows, predicted_rows, gold_width, predicted_width, order_matters, tolerance
        )
        return DatasetScore(Outcome.WRONG_COLUMNS, diagnostics=labels)
    if gold_rows and not predicted_rows:
        return DatasetScore(Outcome.EMPTY_RESULT)

    rules = match_rules(order_matters, tolerance)
    order = find_column_order(gold_rows, predicted_rows, gold_width, gold_width, rules)
    if order is None:
        labels = diagnose(
            gold_rows, predicted_rows, gold_width, gold_width, order_matters, tolerance
        )
        return DatasetScore(Outcome.WRONG_RESULT, diagnostics=labels)
    return DatasetScore(Outcome.CORRECT, order)


def combine(main: DatasetScore, edge: DatasetScore) -> QuestionScore:
    """Correct only if both datasets are; otherwise main's failure, then edge's."""
    if main.outcome == Outcome.CORRECT and edge.outcome == Outcome.CORRECT:
        outcome = Outcome.CORRECT
    elif main.outcome != Outcome.CORRECT:
        outcome = main.outcome
    else:
        outcome = edge.outcome
    lucky_match = main.outcome == Outcome.CORRECT and edge.outcome != Outcome.CORRECT
    return QuestionScore(main, edge, outcome, lucky_match)
