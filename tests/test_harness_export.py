"""Exporting a run: completeness checks, the Parquet file and the run manifest."""

import json
from dataclasses import replace
from pathlib import Path

import pyarrow.parquet as pq
import pytest
from test_harness_store import ROW, RUN

from sqlbench.harness.export import ExportError, export_run
from sqlbench.harness.store import ResultStore
from sqlbench.models.config import ModelConfig, Provider


def model(model_id: str) -> ModelConfig:
    return ModelConfig(
        id=model_id,
        provider=Provider.OPENAI,
        model=model_id,
        version=f"{model_id}-1",
        max_tokens=100,
        temperature=0,
        input_usd_per_million=1.0,
        output_usd_per_million=10.0,
        requests_per_minute=60,
    )


MODELS = [model("model-a"), model("model-b")]


def frozen(tmp_path: Path, question_ids: list[str]) -> Path:
    path = tmp_path / "MANIFEST.json"
    manifest = {
        "scorer_version": 1,
        "prompt": {"file": "prompts/v1.txt", "sha256": "p" * 64},
        "question_set_sha256": "q" * 64,
        "question_ids": question_ids,
        "data_manifest_sha256": "d" * 64,
        "data": {"shop": {"main": "m" * 64, "edge": "e" * 64}},
    }
    path.write_text(json.dumps(manifest))
    return path


def full_store() -> ResultStore:
    """run-1 with every (model, question) stored; model-b's answer came from the cache."""
    store = ResultStore(":memory:")
    store.create_run(RUN)
    store.add_result(ROW)
    store.add_result(replace(ROW, model_id="model-b", cost_usd=0.0, response_cached=True))
    return store


def test_export_writes_every_row_and_a_manifest(tmp_path: Path) -> None:
    exported = export_run(
        full_store(), "run-1", MODELS, frozen(tmp_path, ["shop_001"]), tmp_path / "out"
    )

    table = pq.read_table(exported.parquet_path)
    assert table.num_rows == 2
    assert table.column("model_id").to_pylist() == ["model-a", "model-b"]
    manifest = json.loads(exported.manifest_path.read_text())
    assert manifest["rows"] == 2
    assert manifest["questions"] == 1
    assert manifest["cost_usd_spent"] == pytest.approx(0.0123)
    # Both answers at list price, including the cached one: 2000 in at $1, 50 out at $10.
    assert manifest["cost_usd_at_list_price"] == pytest.approx(2 * 0.0025)
    assert manifest["models"][1]["responses_from_cache"] == 1
    assert manifest["models"][0]["served_models"] == ["model-a-1 via host"]
    assert manifest["finished_at_utc"] >= manifest["started_at_utc"]


def test_export_refuses_a_run_with_missing_rows(tmp_path: Path) -> None:
    with pytest.raises(ExportError, match="2 rows missing, e.g. model-a/snowflake/shop_002"):
        export_run(
            full_store(),
            "run-1",
            MODELS,
            frozen(tmp_path, ["shop_001", "shop_002"]),
            tmp_path / "out",
        )


def test_export_refuses_a_run_on_other_inputs(tmp_path: Path) -> None:
    path = frozen(tmp_path, ["shop_001"])
    manifest = json.loads(path.read_text())
    manifest["prompt"]["sha256"] = "x" * 64
    path.write_text(json.dumps(manifest))

    with pytest.raises(ExportError, match="prompt_sha256 differs"):
        export_run(full_store(), "run-1", MODELS, path, tmp_path / "out")
