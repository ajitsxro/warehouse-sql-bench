import re
import subprocess
import sys
from pathlib import Path

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
