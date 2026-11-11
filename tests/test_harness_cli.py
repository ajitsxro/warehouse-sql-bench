"""`bench run` and `bench rerun-gold` end to end: a full dry run on a local DuckDB copy."""

import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from question_files import write_question
from typer.testing import CliRunner

from sqlbench import cli
from sqlbench.datagen.generator import generate_dataset
from sqlbench.datagen.tables import Profile
from sqlbench.harness.store import ResultStore
from sqlbench.load.dataset import load_dataset
from sqlbench.load.duckdb_loader import DuckDBLoader

FIXTURES = Path(__file__).parent / "fixtures"
SCHEMA = FIXTURES / "mini_shop.yaml"
ORDERED = "SELECT customer_id FROM customers ORDER BY customer_id LIMIT 3"


@pytest.fixture(scope="module")
def loaded(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """mini_shop main and edge, generated and loaded into data/sqlbench.duckdb."""
    root = tmp_path_factory.mktemp("harness")
    data_dir = root / "data"
    manifest_dir = root / "manifests"
    loader = DuckDBLoader(data_dir / "sqlbench.duckdb")
    for profile in (Profile.MAIN, Profile.EDGE):
        generate_dataset(SCHEMA, profile, 5, data_dir, manifest_dir)
        load_dataset(loader, SCHEMA, profile, data_dir, manifest_dir)
    loader.close()
    return root


@pytest.fixture
def workspace(loaded: Path, tmp_path: Path) -> dict[str, Path]:
    questions_dir = tmp_path / "questions"
    write_question(questions_dir)
    write_question(
        questions_dir,
        id="mini_shop_002",
        question="Which are the first three customer ids?",
        gold_sql={"snowflake": ORDERED, "databricks": ORDERED},
        order_matters=True,
    )
    return {"root": loaded, "questions": questions_dir, "results": tmp_path / "results.duckdb"}


def paths(workspace: dict[str, Path]) -> list[str]:
    return [
        "--results-file",
        str(workspace["results"]),
        "--dir",
        str(workspace["questions"]),
        "--data-dir",
        str(workspace["root"] / "data"),
        "--manifest-dir",
        str(workspace["root"] / "manifests"),
    ]


def bench(*arguments: str, answer: str = "") -> Any:
    return CliRunner().invoke(cli.app, list(arguments), input=answer)


def dry_run(workspace: dict[str, Path], *extra: str) -> Any:
    return bench(
        "run",
        "--dry-run",
        "--questions",
        "drafts",
        "--schemas-dir",
        str(FIXTURES),
        *paths(workspace),
        *extra,
    )


def test_a_full_dry_run_scores_and_stores_every_task(workspace: dict[str, Path]) -> None:
    result = dry_run(workspace, "--run-id", "dry-1")

    assert result.exit_code == 0, result.output
    assert "run dry-1: 4 of 4 tasks to do" in result.output
    assert "stored 4 results, spent $0.0000" in result.output
    assert "outcomes so far: correct 4" in result.output
    store = ResultStore(workspace["results"])
    rows = store.results("dry-1")
    assert {(row.warehouse, row.question_id) for row in rows} == {
        ("snowflake", "mini_shop_001"),
        ("snowflake", "mini_shop_002"),
        ("databricks", "mini_shop_001"),
        ("databricks", "mini_shop_002"),
    }
    assert all(row.model_id == "dry-run" and row.scorer_version == 1 for row in rows)
    assert len(store.all_gold()) == 8
    store.close()


def test_resuming_a_finished_run_does_nothing(workspace: dict[str, Path]) -> None:
    dry_run(workspace, "--run-id", "dry-2")

    result = dry_run(workspace, "--run-id", "dry-2")

    assert result.exit_code == 0, result.output
    assert "run dry-2: 0 of 4 tasks to do" in result.output


def test_resuming_with_other_questions_is_refused(workspace: dict[str, Path]) -> None:
    dry_run(workspace, "--run-id", "dry-3")

    result = dry_run(workspace, "--run-id", "dry-3", "--limit", "1")

    assert result.exit_code == 1
    assert "start a new run instead" in result.output


def test_only_verified_questions_run_by_default(workspace: dict[str, Path]) -> None:
    result = bench("run", "--dry-run", "--schemas-dir", str(FIXTURES), *paths(workspace))

    assert result.exit_code == 1
    assert "no verified questions to run" in result.output


def test_rerun_gold_confirms_the_cached_hashes(workspace: dict[str, Path]) -> None:
    dry_run(workspace, "--run-id", "dry-4")

    result = bench("rerun-gold", "--dry-run", *paths(workspace))

    assert result.exit_code == 0, result.output
    assert "match 8, within tolerance 0, mismatch 0, stale 0" in result.output


def test_rerun_gold_reports_changed_gold_as_stale(workspace: dict[str, Path]) -> None:
    dry_run(workspace, "--run-id", "dry-5")
    changed = workspace["questions"] / "mini_shop" / "mini_shop_002.yaml"
    changed.write_text(changed.read_text().replace("LIMIT 3", "LIMIT 2"))

    result = bench("rerun-gold", "--dry-run", *paths(workspace))

    assert "match 4, within tolerance 0, mismatch 0, stale 4" in result.output


def test_a_real_run_shows_the_cost_and_waits(
    workspace: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    models_file = tmp_path / "models.yaml"
    entry = {
        "id": "paid-1",
        "provider": "openai",
        "model": "m",
        "version": "m-1",
        "max_tokens": 1000,
        "temperature": 0,
        "input_usd_per_million": 2.0,
        "output_usd_per_million": 10.0,
        "requests_per_minute": 60,
    }
    models_file.write_text(yaml.safe_dump([entry]))

    def no_calls(*args: Any) -> Any:
        raise AssertionError("nothing may be called before the go")

    monkeypatch.setattr(cli, "make_client", no_calls)
    monkeypatch.setattr(cli, "open_warehouse", no_calls)

    result = bench(
        "run",
        "--questions",
        "drafts",
        "--models",
        "paid-1",
        "--run-id",
        "paid",
        "--models-file",
        str(models_file),
        "--cache-file",
        str(tmp_path / "cache.duckdb"),
        "--schemas-dir",
        str(FIXTURES),
        *paths(workspace),
        answer="n\n",
    )

    assert result.exit_code == 1
    assert "paid-1: 4 model calls, at most $0.05" in result.output
    assert "snowflake: 8 queries" in result.output
    assert "databricks: 8 queries" in result.output
    assert "Start the run?" in result.output


def test_unknown_warehouses_are_rejected(workspace: dict[str, Path]) -> None:
    result = bench("run", "--dry-run", "--warehouses", "duckdb", *paths(workspace))

    assert result.exit_code == 2


def test_freeze_lists_verified_questions_and_the_run_hashes(
    workspace: dict[str, Path], tmp_path: Path
) -> None:
    write_question(
        workspace["questions"],
        id="mini_shop_003",
        status="verified",
        verified_by="tester",
        verified_at="2026-11-10",
    )
    output = tmp_path / "MANIFEST.json"

    result = bench(
        "questions",
        "freeze",
        "--output",
        str(output),
        "--dir",
        str(workspace["questions"]),
        "--schemas-dir",
        str(FIXTURES),
        "--manifest-dir",
        str(workspace["root"] / "manifests"),
    )

    assert result.exit_code == 0, result.output
    assert "froze 1 verified questions" in result.output
    manifest = json.loads(output.read_text())
    assert manifest["question_ids"] == ["mini_shop_003"]
    assert manifest["scorer_version"] == 1
    assert set(manifest["data"]["mini_shop"]) == {"main", "edge"}

    bench(
        "run", "--dry-run", "--run-id", "frozen", "--schemas-dir", str(FIXTURES), *paths(workspace)
    )
    run = ResultStore(workspace["results"]).get_run("frozen")
    assert run is not None
    assert run.question_set_sha256 == manifest["question_set_sha256"]
    assert run.prompt_sha256 == manifest["prompt"]["sha256"]
    assert run.data_manifest_sha256 == manifest["data_manifest_sha256"]


def test_freeze_refuses_when_nothing_is_verified(
    workspace: dict[str, Path], tmp_path: Path
) -> None:
    result = bench(
        "questions",
        "freeze",
        "--output",
        str(tmp_path / "MANIFEST.json"),
        "--dir",
        str(workspace["questions"]),
        "--schemas-dir",
        str(FIXTURES),
    )

    assert result.exit_code == 1
    assert "no verified questions to freeze" in result.output
