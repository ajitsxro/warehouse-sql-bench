"""The hand-verification loop for draft questions.

Each draft is linted and checked, then shown with its golds and the four
result previews. The reviewer verifies, edits, rejects or skips it. Verify is
only offered when lint and the gold check both pass. Prompts, the editor
and output are passed in, so tests can script a review session.
"""

import datetime
import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from sqlbench.datagen.tables import Profile
from sqlbench.questions.check import (
    GOLD_WAREHOUSES,
    PROFILES,
    GoldCheck,
    GoldTargets,
    check_question,
)
from sqlbench.questions.lint import lint_files
from sqlbench.questions.model import Question, Status, load_question
from sqlbench.warehouses.base import WarehouseName

PREVIEW_ROWS = 5
PREVIEW_VALUE_CHARS = 60


class Choice(StrEnum):
    VERIFY = "v"
    EDIT = "e"
    REJECT = "r"
    SKIP = "s"
    QUIT = "q"


@dataclass(frozen=True)
class Inspection:
    """Lint problems, or the question and its gold check when lint passed."""

    problems: list[str]
    question: Question | None = None
    check: GoldCheck | None = None

    @property
    def ready(self) -> bool:
        return not self.problems and self.check is not None and self.check.passed


@dataclass(frozen=True)
class ReviewTools:
    inspect: Callable[[Path], Inspection]
    choose: Callable[[list[Choice]], Choice]
    edit: Callable[[Path], None]
    show: Callable[[str], None]


def inspect_question(path: Path, schemas_dir: Path, targets: GoldTargets) -> Inspection:
    problems = lint_files([path], schemas_dir)[path]
    if problems:
        return Inspection(problems)
    question = load_question(path)
    return Inspection([], question, check_question(question, targets))


def set_status(
    path: Path,
    status: Status,
    verified_by: str | None = None,
    verified_at: datetime.date | None = None,
) -> None:
    """Rewrite the status lines in place, leaving the rest of the file as written.

    Top-level keys start at column 0, and gold SQL is indented under gold_sql,
    so a line match can't touch the SQL. A missing key is appended; a key
    with no value is removed.
    """
    values: dict[str, str | None] = {
        "status": status.value,
        # JSON strings are valid YAML and quote anything a name might contain.
        "verified_by": json.dumps(verified_by) if verified_by is not None else None,
        "verified_at": verified_at.isoformat() if verified_at is not None else None,
    }
    text = path.read_text()
    for key, value in values.items():
        pattern = re.compile(rf"^{key}:.*\n?", re.MULTILINE)
        if value is None:
            text = pattern.sub("", text)
        elif pattern.search(text):
            text = pattern.sub(f"{key}: {value}\n", text)
        else:
            if not text.endswith("\n"):
                text += "\n"
            text += f"{key}: {value}\n"
    path.write_text(text)
    # Fails loudly if the edit somehow broke the file.
    load_question(path)


def review(path: Path, tools: ReviewTools, verifier: str, today: datetime.date) -> Choice:
    """Show one draft until the reviewer decides; edits re-inspect it."""
    while True:
        inspection = tools.inspect(path)
        for line in describe(path, inspection):
            tools.show(line)
        allowed = [Choice.EDIT, Choice.REJECT, Choice.SKIP, Choice.QUIT]
        if inspection.ready:
            allowed.insert(0, Choice.VERIFY)
        choice = tools.choose(allowed)
        if choice == Choice.EDIT:
            tools.edit(path)
            continue
        if choice == Choice.VERIFY:
            set_status(path, Status.VERIFIED, verifier, today)
        elif choice == Choice.REJECT:
            set_status(path, Status.REJECTED)
        return choice


def review_drafts(
    paths: list[Path], tools: ReviewTools, verifier: str, today: datetime.date
) -> dict[Choice, int]:
    """Review each file in turn until quit; counts of each decision."""
    counts = {choice: 0 for choice in Choice}
    for path in paths:
        choice = review(path, tools, verifier, today)
        counts[choice] += 1
        if choice == Choice.QUIT:
            break
    return counts


def describe(path: Path, inspection: Inspection) -> list[str]:
    lines = ["", f"== {path}"]
    question = inspection.question
    if question is None:
        lines.append("Lint problems (fix them with edit):")
        for problem in inspection.problems:
            lines.append(f"  {problem}")
        return lines

    lines.append(f"Question: {question.question}")
    category = f"{question.category} / {question.divergence}"
    if question.divergence_note:
        category += f": {question.divergence_note}"
    lines.append(f"Category: {category}")
    lines.append(
        f"Features: {', '.join(question.features)}; {question.difficulty}; "
        f"order_matters={question.order_matters}; float_tolerance={question.float_tolerance}"
    )
    for warehouse in GOLD_WAREHOUSES:
        lines.append(f"{warehouse} gold:")
        for sql_line in question.gold_sql.for_warehouse(warehouse).strip().splitlines():
            lines.append(f"    {sql_line}")

    check = inspection.check
    assert check is not None
    for warehouse in GOLD_WAREHOUSES:
        for profile in PROFILES:
            lines.extend(preview(f"{warehouse}/{profile}", check, warehouse, profile))
    if check.passed:
        lines.append("Check: passed")
    else:
        lines.append("Check: failed")
        for failure in check.failures:
            lines.append(f"  {failure}")
    return lines


def preview(label: str, check: GoldCheck, warehouse: WarehouseName, profile: Profile) -> list[str]:
    result = check.run(warehouse, profile)
    if result.error is not None:
        return [f"{label}: {result.error.category} error"]
    lines = [f"{label}: {len(result.rows)} rows"]
    for row in result.rows[:PREVIEW_ROWS]:
        values = [short(value) for value in row]
        lines.append(f"    {' | '.join(values)}")
    if len(result.rows) > PREVIEW_ROWS:
        lines.append("    ...")
    return lines


def short(value: Any) -> str:
    text = "NULL" if value is None else str(value).replace("\n", " ")
    if len(text) > PREVIEW_VALUE_CHARS:
        return text[: PREVIEW_VALUE_CHARS - 3] + "..."
    return text
