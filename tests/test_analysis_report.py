"""The generated README and charts, on the hand-built run from test_analysis_tables."""

from pathlib import Path

from test_analysis_tables import MANIFEST, run_data

from sqlbench.analysis.guard_check import GuardCheck
from sqlbench.analysis.hand_labels import HandLabel, RootCause
from sqlbench.analysis.report import ReportInputs, short_error, write_report


def test_errors_lose_request_ids_and_query_echoes() -> None:
    snowflake = "002139 (02000): SQL compilation error:\nUnknown function X. (request_id=abc, q=1)"
    databricks = "[PARSE_SYNTAX_ERROR] near 'FLATTEN'.\n\n== SQL ==\nSELECT 1"

    assert short_error(snowflake) == "002139 (02000): SQL compilation error: Unknown function X."
    assert short_error(databricks) == "[PARSE_SYNTAX_ERROR] near 'FLATTEN'."


def test_report_writes_readme_and_charts(tmp_path: Path) -> None:
    data = run_data()
    guard = [GuardCheck("weak", "databricks", "mini_shop_002", True, "")]
    hand = [
        HandLabel("test", ("weak", "databricks", "mini_shop_002"), RootCause.DIALECT_SYNTAX, "")
    ]

    path = write_report(ReportInputs(data, MANIFEST, guard, hand, "by a test"), tmp_path)

    text = path.read_text()
    for chart in ("failures.png", "accuracy.png", "features.png"):
        assert (tmp_path / chart).stat().st_size > 0
        assert f"({chart})" in text
    assert "| weak | 66.7% [" in text
    assert "| dialect_leak | dialect_syntax | 1 | 1 |" in text
    assert "accepted: 1" in text
    assert "Failures labelled: 1, by a test." in text
    assert "| weak | 1 of 6 | 66.7% | 50.0% | +16.7 pp |" in text


def test_report_says_when_inputs_are_missing(tmp_path: Path) -> None:
    path = write_report(ReportInputs(run_data(), MANIFEST, None, [], "by hand"), tmp_path)

    text = path.read_text()
    assert "Not checked: run `bench analysis guard-check`." in text
    assert "No labels yet" in text
