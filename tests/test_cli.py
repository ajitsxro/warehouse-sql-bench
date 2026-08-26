import subprocess
import sys
from pathlib import Path


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
    assert "Usage: bench" in completed.stdout
