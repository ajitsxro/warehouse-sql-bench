"""Read an exported run (results/run-<id>.parquet) back into ResultRows."""

from dataclasses import fields
from pathlib import Path

import pyarrow.parquet as pq

from sqlbench.harness.store import ResultRow

ROW_FIELDS = [field.name for field in fields(ResultRow)]


def run_parquet_path(run_id: str, results_dir: Path) -> Path:
    return results_dir / f"run-{run_id}.parquet"


def load_run(path: Path) -> list[ResultRow]:
    """Every row of the export, sorted by model, warehouse and question."""
    rows = []
    for record in pq.read_table(path).to_pylist():
        values = {name: record[name] for name in ROW_FIELDS}
        values["features"] = tuple(values["features"])
        values["near_miss"] = tuple(values["near_miss"])
        rows.append(ResultRow(**values))
    rows.sort(key=lambda row: (row.model_id, row.warehouse, row.question_id))
    return rows
