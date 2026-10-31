"""The review loop: status edits, which choices are offered, and `bench questions verify`."""

import datetime
from pathlib import Path
from typing import Any

import pytest
from question_files import FakeWarehouse, as_targets, both_golds, targets, write_question
from typer.testing import CliRunner

from sqlbench import cli
from sqlbench.questions.model import Status, load_question
from sqlbench.questions.verify import (
    Choice,
    ReviewTools,
    inspect_question,
    review,
    review_drafts,
    set_status,
)
from sqlbench.warehouses.base import SessionContext, Warehouse, WarehouseName

FIXTURES = Path(__file__).parent / "fixtures"
TODAY = datetime.date(2026, 11, 10)


class Session:
    """A scripted reviewer: answers in order, records what was offered and shown."""

    def __init__(self, answers: list[Choice]) -> None:
        self.answers = answers
        self.offered: list[list[Choice]] = []
        self.shown: list[str] = []
        self.edits: list[Path] = []
        self.fix_on_edit: dict[str, Any] | None = None
        self.targets = as_targets(targets([(1,)], [(1,)], [(1,)], [(1,)]))

    def choose(self, allowed: list[Choice]) -> Choice:
        self.offered.append(allowed)
        return self.answers.pop(0)

    def edit(self, path: Path) -> None:
        self.edits.append(path)
        if self.fix_on_edit is not None:
            write_question(path.parent.parent, **self.fix_on_edit)

    def tools(self) -> ReviewTools:
        return ReviewTools(
            inspect=lambda path: inspect_question(path, FIXTURES, self.targets),
            choose=self.choose,
            edit=self.edit,
            show=self.shown.append,
        )


def test_verify_stamps_the_reviewer_and_date_and_keeps_the_sql(tmp_path: Path) -> None:
    path = write_question(tmp_path)
    before = path.read_text()

    set_status(path, Status.VERIFIED, "Ajit S", TODAY)

    question = load_question(path)
    assert question.status == Status.VERIFIED
    assert question.verified_by == "Ajit S"
    assert question.verified_at == TODAY
    stamp_keys = ("status:", "verified_by:", "verified_at:")
    kept_before = [line for line in before.splitlines() if not line.startswith(stamp_keys)]
    kept_after = [line for line in path.read_text().splitlines() if not line.startswith(stamp_keys)]
    assert kept_after == kept_before


def test_reject_removes_an_old_verification(tmp_path: Path) -> None:
    path = write_question(tmp_path)
    set_status(path, Status.VERIFIED, "Ajit", TODAY)

    set_status(path, Status.REJECTED)

    text = path.read_text()
    assert "status: rejected" in text
    assert "verified_by" not in text
    assert "verified_at" not in text


def test_status_is_added_when_the_file_has_none(tmp_path: Path) -> None:
    path = write_question(tmp_path)
    path.write_text(path.read_text().replace("status: draft\n", ""))

    set_status(path, Status.REJECTED)

    assert load_question(path).status == Status.REJECTED


def test_a_passing_draft_can_be_verified(tmp_path: Path) -> None:
    path = write_question(tmp_path)
    session = Session([Choice.VERIFY])

    choice = review(path, session.tools(), "Ajit", TODAY)

    assert choice == Choice.VERIFY
    assert session.offered == [
        [Choice.VERIFY, Choice.EDIT, Choice.REJECT, Choice.SKIP, Choice.QUIT]
    ]
    assert load_question(path).verified_by == "Ajit"
    assert "Check: passed" in session.shown
    assert "snowflake/main: 1 rows" in session.shown


def test_verify_is_not_offered_when_the_check_fails(tmp_path: Path) -> None:
    path = write_question(tmp_path)
    session = Session([Choice.SKIP])
    session.targets = as_targets(targets([(1,)], [(2,)], [(1,)], [(1,)]))

    review(path, session.tools(), "Ajit", TODAY)

    assert session.offered == [[Choice.EDIT, Choice.REJECT, Choice.SKIP, Choice.QUIT]]
    assert "Check: failed" in session.shown
    assert load_question(path).status == Status.DRAFT


def test_an_edit_is_inspected_again_before_the_next_choice(tmp_path: Path) -> None:
    path = write_question(tmp_path, gold_sql=both_golds("SELECT * FROM shoppers"))
    session = Session([Choice.EDIT, Choice.VERIFY])
    session.fix_on_edit = {}

    review(path, session.tools(), "Ajit", TODAY)

    assert session.edits == [path]
    assert Choice.VERIFY not in session.offered[0]
    assert Choice.VERIFY in session.offered[1]
    assert "Lint problems (fix them with edit):" in session.shown
    assert load_question(path).status == Status.VERIFIED


def test_reject_and_skip(tmp_path: Path) -> None:
    rejected = write_question(tmp_path)
    skipped = write_question(tmp_path, id="mini_shop_002")
    before = skipped.read_text()

    counts = review_drafts(
        [rejected, skipped], Session([Choice.REJECT, Choice.SKIP]).tools(), "Ajit", TODAY
    )

    assert counts[Choice.REJECT] == 1
    assert counts[Choice.SKIP] == 1
    assert load_question(rejected).status == Status.REJECTED
    assert skipped.read_text() == before


def test_quit_stops_before_the_next_draft(tmp_path: Path) -> None:
    first = write_question(tmp_path)
    second = write_question(tmp_path, id="mini_shop_002")
    session = Session([Choice.QUIT])

    counts = review_drafts([first, second], session.tools(), "Ajit", TODAY)

    assert counts[Choice.QUIT] == 1
    assert len(session.offered) == 1
    assert load_question(second).status == Status.DRAFT


@pytest.fixture
def fake_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    def open_fake(name: WarehouseName, context: SessionContext | None = None) -> Warehouse:
        return FakeWarehouse(name, [(1,)])

    monkeypatch.setattr(cli, "open_warehouse", open_fake)
    monkeypatch.setattr(cli, "git_user_name", lambda: "Ajit")


def run_verify(questions_dir: Path, answers: str) -> Any:
    arguments = ["questions", "verify", "--schema", "mini_shop", "--dir", str(questions_dir)]
    arguments += ["--schemas-dir", str(FIXTURES), "--yes"]
    return CliRunner().invoke(cli.app, arguments, input=answers)


def test_cli_verify_reviews_only_drafts(tmp_path: Path, fake_cli: None) -> None:
    draft = write_question(tmp_path)
    write_question(tmp_path, id="mini_shop_002", status="rejected")

    result = run_verify(tmp_path, "x\nv\n")

    assert result.exit_code == 0, result.output
    assert "choose one of: v, e, r, s, q" in result.output
    assert "mini_shop_002" not in result.output
    assert "verified 1, rejected 0, skipped 0" in result.output
    assert load_question(draft).verified_at == datetime.date.today()


def test_cli_verify_needs_a_git_user_name(
    tmp_path: Path, fake_cli: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_question(tmp_path)
    monkeypatch.setattr(cli, "git_user_name", lambda: "")

    result = run_verify(tmp_path, "v\n")

    assert result.exit_code == 1
    assert "set git config user.name" in result.output
    assert load_question(tmp_path / "mini_shop" / "mini_shop_001.yaml").status == Status.DRAFT
