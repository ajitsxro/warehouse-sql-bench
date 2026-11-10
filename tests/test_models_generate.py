"""Generating SQL with a fake provider: caching, extraction, costs and `bench generate`."""

from pathlib import Path
from typing import Any

import pytest
import yaml
from question_files import write_question
from typer.testing import CliRunner

from sqlbench import cli
from sqlbench.datagen.generator import generate_dataset
from sqlbench.datagen.tables import Profile
from sqlbench.models.cache import ResponseCache
from sqlbench.models.clients import Completion, ProviderError
from sqlbench.models.config import ModelConfig
from sqlbench.models.generate import build_prompt, estimate_cost, generate
from sqlbench.models.prompt import load_template
from sqlbench.questions.model import load_question
from sqlbench.schema.spec import load_schema
from sqlbench.warehouses.base import WarehouseName

ROOT = Path(__file__).parent.parent
FIXTURES = Path(__file__).parent / "fixtures"
TEMPLATE = load_template(ROOT / "prompts" / "v1.txt")
ANSWER = "Here you go:\n```sql\nSELECT segment, COUNT(*) FROM customers GROUP BY segment;\n```"


class FakeProvider:
    """Answers every prompt the same way and counts calls."""

    def __init__(self, text: str = ANSWER, error: ProviderError | None = None) -> None:
        self.text = text
        self.error = error
        self.prompts: list[str] = []

    def complete(self, prompt: str) -> Completion:
        self.prompts.append(prompt)
        if self.error is not None:
            raise self.error
        return Completion(self.text, 2000, 100, "fake-served", "stop")


MODEL_ENTRY: dict[str, Any] = {
    "id": "fake-1",
    "provider": "openai",
    "model": "fake-model",
    "version": "fake-model-2026",
    "max_tokens": 1000,
    "temperature": 0,
    "input_usd_per_million": 2.0,
    "output_usd_per_million": 10.0,
    "requests_per_minute": 6000,
}
MODEL = ModelConfig.model_validate(MODEL_ENTRY)


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("generate")
    generate_dataset(FIXTURES / "mini_shop.yaml", Profile.MAIN, 5, root / "data", root / "m")
    return root / "data"


def test_generate_extracts_sql_and_charges_for_the_call(tmp_path: Path, data_dir: Path) -> None:
    question = load_question(write_question(tmp_path))
    schema = load_schema(FIXTURES / "mini_shop.yaml")
    prompt = build_prompt(TEMPLATE, schema, WarehouseName.SNOWFLAKE, question, data_dir)
    provider = FakeProvider()
    cache = ResponseCache(":memory:")

    first = generate(question, prompt, WarehouseName.SNOWFLAKE, MODEL, provider, cache, TEMPLATE)
    second = generate(question, prompt, WarehouseName.SNOWFLAKE, MODEL, provider, cache, TEMPLATE)

    assert first.sql == "SELECT segment, COUNT(*) FROM customers GROUP BY segment"
    assert first.cost_usd == pytest.approx(2000 * 2.0 / 1e6 + 100 * 10.0 / 1e6)
    assert not first.cached
    assert second.cached
    assert second.cost_usd == 0.0
    assert second.completion == first.completion
    assert len(provider.prompts) == 1
    assert first.template_sha256 == TEMPLATE.sha256
    assert first.model_version == "fake-model-2026"


def test_a_response_without_sql_is_recorded_as_none(tmp_path: Path, data_dir: Path) -> None:
    question = load_question(write_question(tmp_path))
    schema = load_schema(FIXTURES / "mini_shop.yaml")
    prompt = build_prompt(TEMPLATE, schema, WarehouseName.DATABRICKS, question, data_dir)

    result = generate(
        question,
        prompt,
        WarehouseName.DATABRICKS,
        MODEL,
        FakeProvider("I need more context."),
        ResponseCache(":memory:"),
        TEMPLATE,
    )

    assert result.sql is None
    assert result.completion.text == "I need more context."


def test_the_estimate_is_an_upper_bound_on_output() -> None:
    estimate = estimate_cost(MODEL, ["x" * 4000, "y" * 4000])

    assert estimate == pytest.approx((2002 * 2.0 + 2000 * 10.0) / 1e6)
    assert estimate_cost(MODEL, []) == 0.0


@pytest.fixture
def workspace(tmp_path: Path, data_dir: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    models_file = tmp_path / "models.yaml"
    models_file.write_text(yaml.safe_dump([MODEL_ENTRY]))
    provider = FakeProvider()
    monkeypatch.setattr(cli, "make_client", lambda config: provider)
    questions_dir = tmp_path / "questions"
    write_question(questions_dir)
    write_question(questions_dir, id="mini_shop_002", question="Another question?")
    write_question(questions_dir, id="mini_shop_003", status="rejected")
    return {
        "root": tmp_path,
        "provider": provider,
        "models_file": models_file,
        "questions_dir": questions_dir,
        "data_dir": data_dir,
    }


def run_generate(workspace: dict[str, Any], *extra: str, answer: str = "") -> Any:
    arguments = [
        "generate",
        "--model",
        "fake-1",
        "--warehouse",
        "snowflake",
        "--models-file",
        str(workspace["models_file"]),
        "--cache-file",
        str(workspace["root"] / "cache.duckdb"),
        "--dir",
        str(workspace["questions_dir"]),
        "--schemas-dir",
        str(FIXTURES),
        "--data-dir",
        str(workspace["data_dir"]),
        *extra,
    ]
    return CliRunner().invoke(cli.app, arguments, input=answer)


def test_cli_uses_only_verified_questions_by_default(workspace: dict[str, Any]) -> None:
    result = run_generate(workspace, "--yes")

    assert result.exit_code == 1
    assert "no questions to generate for" in result.output
    assert workspace["provider"].prompts == []


def test_cli_spot_check_on_drafts(workspace: dict[str, Any]) -> None:
    result = run_generate(workspace, "--drafts", "--limit", "1", "--yes")

    assert result.exit_code == 0, result.output
    assert "1 questions, 1 not cached" in result.output
    assert "mini_shop_001: 2,000 in / 100 out tokens, $0.0050" in result.output
    assert "  SELECT segment, COUNT(*) FROM customers GROUP BY segment" in result.output
    assert len(workspace["provider"].prompts) == 1


def test_cli_second_run_is_free(workspace: dict[str, Any]) -> None:
    run_generate(workspace, "--drafts", "--yes")
    result = run_generate(workspace, "--drafts")

    assert result.exit_code == 0, result.output
    assert "2 questions, 0 not cached: at most $0.00" in result.output
    assert "Call the model?" not in result.output
    assert result.output.count("cached") >= 2
    assert len(workspace["provider"].prompts) == 2


def test_cli_stops_when_the_cost_is_declined(workspace: dict[str, Any]) -> None:
    result = run_generate(workspace, "--drafts", answer="n\n")

    assert result.exit_code == 1
    assert "Call the model?" in result.output
    assert workspace["provider"].prompts == []


def test_cli_reports_provider_errors_and_continues(
    workspace: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    failing = FakeProvider(error=ProviderError("400 bad request", retryable=False))
    monkeypatch.setattr(cli, "make_client", lambda config: failing)

    result = run_generate(workspace, "--drafts", "--yes")

    assert result.exit_code == 1
    assert "mini_shop_001: provider error: 400 bad request" in result.output
    assert len(failing.prompts) == 2
    assert "2 provider errors" in result.output


def test_cli_rejects_duckdb(workspace: dict[str, Any]) -> None:
    arguments = ["generate", "--model", "fake-1", "--warehouse", "duckdb"]

    assert CliRunner().invoke(cli.app, arguments).exit_code == 2
