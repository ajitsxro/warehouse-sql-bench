"""Export a finished run: its rows as Parquet and a manifest of what produced them.

Export refuses a run that is missing rows or doesn't match the frozen
inputs, so a committed results file always covers the whole frozen set.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq

from sqlbench.datagen.generator import file_sha256
from sqlbench.harness.store import ResultRow, ResultStore, RunInfo
from sqlbench.models.config import ModelConfig, find_model


class ExportError(Exception):
    pass


@dataclass(frozen=True)
class ExportedRun:
    parquet_path: Path
    manifest_path: Path
    rows: int
    cost_usd: float


def check_complete(run: RunInfo, rows: list[ResultRow], frozen: dict[str, object]) -> None:
    """Raise ExportError unless the run has one row per (model, warehouse, frozen question)."""
    for name in ("question_set_sha256", "data_manifest_sha256", "scorer_version"):
        if getattr(run, name) != frozen[name]:
            raise ExportError(f"run {run.run_id}: {name} differs from the frozen manifest")
    prompt = frozen["prompt"]
    if not isinstance(prompt, dict) or run.prompt_sha256 != prompt["sha256"]:
        raise ExportError(f"run {run.run_id}: prompt_sha256 differs from the frozen manifest")
    question_ids = frozen["question_ids"]
    if not isinstance(question_ids, list):
        raise ExportError("frozen manifest has no question_ids list")
    stored = {(row.model_id, row.warehouse, row.question_id) for row in rows}
    missing = []
    for model_id in run.models:
        for warehouse in run.warehouses:
            for question_id in question_ids:
                if (model_id, warehouse, question_id) not in stored:
                    missing.append(f"{model_id}/{warehouse}/{question_id}")
    if missing:
        shown = ", ".join(missing[:5])
        raise ExportError(f"run {run.run_id}: {len(missing)} rows missing, e.g. {shown}")


def model_summary(model: ModelConfig, rows: list[ResultRow]) -> dict[str, object]:
    """Config, what the provider reported serving, and cost for one model's rows."""
    served = sorted({row.served_model for row in rows if row.served_model})
    spent = 0.0
    at_list_price = 0.0
    cached = 0
    for row in rows:
        spent += row.cost_usd
        at_list_price += model.cost_usd(row.input_tokens, row.output_tokens)
        if row.response_cached:
            cached += 1
    return {
        "id": model.id,
        "provider": model.provider.value,
        "model": model.model,
        "version": model.version,
        "served_models": served,
        "temperature": model.temperature,
        "max_tokens": model.max_tokens,
        "input_usd_per_million": model.input_usd_per_million,
        "output_usd_per_million": model.output_usd_per_million,
        "responses": len(rows),
        "responses_from_cache": cached,
        "cost_usd_spent": round(spent, 6),
        "cost_usd_at_list_price": round(at_list_price, 6),
    }


def export_run(
    store: ResultStore,
    run_id: str,
    models: list[ModelConfig],
    frozen_path: Path,
    output_dir: Path,
) -> ExportedRun:
    """Write run-<id>.parquet and run-<id>.manifest.json into output_dir."""
    run = store.get_run(run_id)
    if run is None:
        raise ExportError(f"no run {run_id}")
    rows = store.results(run_id)
    frozen = json.loads(frozen_path.read_text())
    check_complete(run, rows, frozen)

    output_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = output_dir / f"run-{run_id}.parquet"
    pq.write_table(store.results_table(run_id), parquet_path, compression="zstd")

    summaries = []
    total_spent = 0.0
    total_at_list_price = 0.0
    for model_id in run.models:
        model = find_model(models, model_id)
        model_rows = [row for row in rows if row.model_id == model_id]
        summaries.append(model_summary(model, model_rows))
        for row in model_rows:
            total_spent += row.cost_usd
            total_at_list_price += model.cost_usd(row.input_tokens, row.output_tokens)
    started_at, finished_at = store.run_span(run_id)
    manifest = {
        "run_id": run.run_id,
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": finished_at.isoformat() if finished_at else None,
        "git_sha": run.git_sha,
        "prompt": frozen["prompt"],
        "question_set_sha256": run.question_set_sha256,
        "questions": len(frozen["question_ids"]),
        "data_manifest_sha256": run.data_manifest_sha256,
        "data": frozen["data"],
        "scorer_version": run.scorer_version,
        "frozen_manifest_sha256": file_sha256(frozen_path),
        "warehouses": list(run.warehouses),
        "models": summaries,
        "rows": len(rows),
        "cost_usd_spent": round(total_spent, 6),
        "cost_usd_at_list_price": round(total_at_list_price, 6),
        "parquet_sha256": file_sha256(parquet_path),
    }
    manifest_path = output_dir / f"run-{run_id}.manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return ExportedRun(parquet_path, manifest_path, len(rows), total_spent)
