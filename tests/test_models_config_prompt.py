"""models.yaml parsing, cost accounting, and the rendered prompt."""

import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
import yaml

from sqlbench.datagen.generator import generate_dataset
from sqlbench.datagen.tables import Profile
from sqlbench.models.config import ModelConfig, Provider, find_model, load_models
from sqlbench.models.prompt import format_value, load_template, render_prompt
from sqlbench.schema.spec import load_schema
from sqlbench.warehouses.base import WarehouseName

ROOT = Path(__file__).parent.parent
FIXTURE = Path(__file__).parent / "fixtures" / "mini_shop.yaml"


def model_entry(**changes: Any) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "id": "fake-1",
        "provider": "openai",
        "model": "fake-model",
        "version": "fake-model-2026",
        "max_tokens": 1000,
        "temperature": 0,
        "input_usd_per_million": 2.0,
        "output_usd_per_million": 10.0,
        "requests_per_minute": 60,
    }
    entry.update(changes)
    return entry


def test_the_committed_models_file_lists_six_models() -> None:
    models = load_models(ROOT / "models.yaml")

    assert len(models) == 6
    providers = {model.provider for model in models}
    assert providers == set(Provider)


def test_cost_uses_the_per_million_prices() -> None:
    model = ModelConfig.model_validate(model_entry())

    assert model.cost_usd(1_000_000, 0) == pytest.approx(2.0)
    assert model.cost_usd(5000, 1000) == pytest.approx(0.01 + 0.01)


def test_temperature_may_be_left_unset() -> None:
    assert ModelConfig.model_validate(model_entry(temperature=None)).temperature is None


@pytest.mark.parametrize(
    "changes",
    [{"id": "Has Spaces"}, {"max_tokens": 0}, {"temperature": -1}, {"provider": "acme"}, {"x": 1}],
)
def test_invalid_entries_are_rejected(changes: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        ModelConfig.model_validate(model_entry(**changes))


def test_duplicate_ids_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "models.yaml"
    path.write_text(yaml.safe_dump([model_entry(), model_entry()]))

    with pytest.raises(ValueError, match="fake-1 is listed twice"):
        load_models(path)


def test_unknown_model_ids_name_the_known_ones() -> None:
    models = [ModelConfig.model_validate(model_entry())]

    with pytest.raises(KeyError, match="known: fake-1"):
        find_model(models, "nope")


def test_template_hash_changes_with_its_text(tmp_path: Path) -> None:
    first = tmp_path / "a.txt"
    second = tmp_path / "b.txt"
    first.write_text("Question: {question}")
    second.write_text("Question: {question}!")

    assert load_template(first).sha256 != load_template(second).sha256
    assert load_template(first).sha256 == load_template(first).sha256


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("prompt")
    generate_dataset(FIXTURE, Profile.MAIN, 5, root / "data", root / "manifests")
    return root / "data"


@pytest.mark.parametrize(
    ("warehouse", "label", "type_name"),
    [
        (WarehouseName.SNOWFLAKE, "Snowflake SQL dialect", "VARIANT"),
        (WarehouseName.DATABRICKS, "Databricks SQL dialect", "MAP<STRING, STRING>"),
    ],
)
def test_prompt_has_the_warehouse_ddl_samples_and_question(
    data_dir: Path, warehouse: WarehouseName, label: str, type_name: str
) -> None:
    template = load_template(ROOT / "prompts" / "v1.txt")
    prompt = render_prompt(template, load_schema(FIXTURE), warehouse, "How many orders?", data_dir)

    assert label in prompt
    assert type_name in prompt
    assert "CREATE OR REPLACE TABLE orders (" in prompt
    assert "REFERENCES customers (customer_id)" in prompt
    assert "mini_shop." not in prompt
    assert "Question: How many orders?" in prompt
    assert prompt.rstrip().endswith("Return one SQL query in a ```sql block.")


def test_prompt_shows_three_rows_per_table(data_dir: Path) -> None:
    template = load_template(ROOT / "prompts" / "v1.txt")
    prompt = render_prompt(template, load_schema(FIXTURE), WarehouseName.SNOWFLAKE, "q", data_dir)
    samples = prompt.split("Sample rows (the first 3 rows of each table):")[1].split("Question:")[0]
    blocks = [block for block in samples.strip().split("\n\n") if block]

    assert [block.splitlines()[0] for block in blocks] == ["customers", "orders", "events"]
    for block in blocks:
        assert len(block.splitlines()) == 2 + 3


@pytest.mark.parametrize(
    ("value", "text"),
    [
        (None, "NULL"),
        (True, "true"),
        (Decimal("12.50"), "12.50"),
        (datetime.date(2024, 2, 29), "2024-02-29"),
        (datetime.datetime(2024, 3, 10, 7, 0, tzinfo=datetime.UTC), "2024-03-10 07:00:00+00:00"),
        ([("size", "L")], '{"size": "L"}'),
        (["a", "b"], '["a", "b"]'),
        ('{"a":1}', '{"a":1}'),
    ],
)
def test_sample_values_are_formatted_for_reading(value: Any, text: str) -> None:
    assert format_value(value) == text
