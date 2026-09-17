import re
import subprocess
import sys
from pathlib import Path

from typer.testing import CliRunner

from sqlbench.cli import app

# Typer forces colored output when GITHUB_ACTIONS or FORCE_COLOR is set, which
# puts escape codes between words, so compare against the plain text.
ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")


def test_bench_help_runs_installed_entry_point() -> None:
    # Call the console script rather than the Typer app object so the test
    # also catches a broken [project.scripts] entry in pyproject.toml.
    bench_path = Path(sys.executable).parent / "bench"
    completed = subprocess.run(
        [str(bench_path), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    plain_output = ANSI_ESCAPE.sub("", completed.stdout)
    assert "Usage: bench" in plain_output


def test_ping_duckdb_prints_latency() -> None:
    result = CliRunner().invoke(app, ["ping", "--warehouse", "duckdb"])

    assert result.exit_code == 0, result.output
    assert re.fullmatch(r"duckdb: ok, SELECT 1 took \d+ ms\n", result.output)


def test_ping_rejects_unknown_warehouse() -> None:
    result = CliRunner().invoke(app, ["ping", "--warehouse", "postgres"])

    assert result.exit_code == 2


def test_schema_validate_reports_ok_and_counts(tmp_path: Path) -> None:
    fixture = Path(__file__).parent / "fixtures" / "mini_shop.yaml"
    (tmp_path / "mini_shop.yaml").write_text(fixture.read_text())

    result = CliRunner().invoke(app, ["schema", "validate", "--dir", str(tmp_path)])

    assert result.exit_code == 0, result.output
    assert "mini_shop: ok (3 tables, 2,800 main rows, 1,020 edge rows)" in result.output


def test_schema_validate_fails_on_any_invalid_file(tmp_path: Path) -> None:
    fixture = Path(__file__).parent / "fixtures" / "mini_shop.yaml"
    (tmp_path / "mini_shop.yaml").write_text(fixture.read_text())
    (tmp_path / "broken.yaml").write_text("name: broken\ndescription: x\ntables: []\n")

    result = CliRunner().invoke(app, ["schema", "validate", "--dir", str(tmp_path)])

    assert result.exit_code == 1
    assert "broken: invalid" in result.output
    assert "mini_shop: ok" in result.output
