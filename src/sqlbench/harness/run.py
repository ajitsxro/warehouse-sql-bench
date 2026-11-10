"""Run every (model, warehouse, question) task, resumably, with a worker pool per warehouse.

A run is identified by its run_id and fingerprinted by what its results
depend on (git sha, prompt, question set, data, scorer version). Resuming
a run skips the tasks already stored, and refuses if the fingerprint
changed, so one run never mixes two setups.
"""

import hashlib
import subprocess
import threading
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

from sqlbench.datagen.generator import file_sha256
from sqlbench.datagen.tables import Profile
from sqlbench.harness.executor import Task, data_sha256, ensure_gold, execute_task
from sqlbench.harness.store import ResultRow, ResultStore, RunInfo, StoredGold
from sqlbench.models.cache import ResponseCache
from sqlbench.models.clients import ModelClient, ProviderError
from sqlbench.models.config import ModelConfig
from sqlbench.models.generate import build_prompt
from sqlbench.models.prompt import PromptTemplate
from sqlbench.questions.check import PROFILES
from sqlbench.questions.model import Question
from sqlbench.schema.spec import SchemaSpec
from sqlbench.scoring.outcomes import GoldError
from sqlbench.warehouses.base import Warehouse, WarehouseName

DEFAULT_WORKERS_PER_WAREHOUSE = 4

# Opens a read-only adapter for (warehouse, schema name, dataset).
AdapterFactory = Callable[[WarehouseName, str, Profile], Warehouse]


class RunError(Exception):
    """The run can't start or resume as asked."""


def git_sha(repo_dir: Path) -> str:
    """HEAD's sha, marked -dirty when tracked files have uncommitted changes."""
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo_dir, capture_output=True, text=True, check=True
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=repo_dir,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return f"{head}-dirty" if status else head


def question_set_sha256(paths: list[Path]) -> str:
    """Hash of the question files in the run: ids and exact file contents."""
    parts = sorted(f"{path.stem}:{file_sha256(path)}" for path in paths)
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()


def data_manifest_sha256(manifest_dir: Path, schema_names: list[str]) -> str:
    parts = []
    for schema_name in sorted(set(schema_names)):
        for profile in PROFILES:
            parts.append(
                f"{profile}/{schema_name}:{data_sha256(manifest_dir, schema_name, profile)}"
            )
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()


def sample_per_schema(questions: list[Question], per_schema: int) -> list[Question]:
    """per_schema questions from each schema, picked by the hash of their id.

    Ids are numbered by category (portable first), so the first N would be
    lopsided. Hash order mixes categories and is the same on every machine.
    """
    by_schema: dict[str, list[Question]] = {}
    for question in questions:
        by_schema.setdefault(question.schema_name, []).append(question)
    picked = []
    for schema_questions in by_schema.values():
        ranked = sorted(schema_questions, key=lambda q: hashlib.sha256(q.id.encode()).hexdigest())
        picked.extend(ranked[:per_schema])
    return sorted(picked, key=lambda q: q.id)


def build_tasks(
    models: list[ModelConfig],
    warehouses: list[WarehouseName],
    questions: list[Question],
    schemas: dict[str, SchemaSpec],
    template: PromptTemplate,
    data_dir: Path,
) -> list[Task]:
    """Every (model, warehouse, question); the prompt depends on warehouse and question only."""
    tasks = []
    for warehouse in warehouses:
        for question in questions:
            schema = schemas[question.schema_name]
            prompt = build_prompt(template, schema, warehouse, question, data_dir)
            for model in models:
                tasks.append(Task(model, warehouse, question, prompt))
    return tasks


def start_or_resume(store: ResultStore, run: RunInfo) -> None:
    existing = store.get_run(run.run_id)
    if existing is None:
        store.create_run(run)
        return
    if existing.fingerprint() != run.fingerprint():
        raise RunError(
            f"run {run.run_id} was started with a different git sha, prompt, question set, "
            "data or scorer version; start a new run instead"
        )
    if (existing.models, existing.warehouses) != (run.models, run.warehouses):
        raise RunError(f"run {run.run_id} covers models {existing.models} on {existing.warehouses}")


def pending(store: ResultStore, run_id: str, tasks: list[Task]) -> list[Task]:
    done = store.completed(run_id)
    return [task for task in tasks if task.key not in done]


@dataclass
class RunSummary:
    stored: int = 0
    cost_usd: float = 0.0
    failures: list[str] = field(default_factory=list)


class WorkerAdapters:
    """Each worker thread opens its own adapters; connections aren't shared across threads."""

    def __init__(self, factory: AdapterFactory) -> None:
        self._factory = factory
        self._local = threading.local()
        self._opened: list[Warehouse] = []
        self._lock = threading.Lock()

    def get(self, warehouse: WarehouseName, schema_name: str, profile: Profile) -> Warehouse:
        if not hasattr(self._local, "adapters"):
            self._local.adapters = {}
        adapters: dict[tuple[WarehouseName, str, Profile], Warehouse] = self._local.adapters
        key = (warehouse, schema_name, profile)
        if key not in adapters:
            adapter = self._factory(warehouse, schema_name, profile)
            adapters[key] = adapter
            with self._lock:
                self._opened.append(adapter)
        return adapters[key]

    def close_all(self) -> None:
        with self._lock:
            for adapter in self._opened:
                adapter.close()
            self._opened.clear()


def run_tasks(
    tasks: list[Task],
    run: RunInfo,
    store: ResultStore,
    cache: ResponseCache,
    template: PromptTemplate,
    clients: dict[str, ModelClient],
    open_adapter: AdapterFactory,
    manifest_dir: Path,
    workers_per_warehouse: int = DEFAULT_WORKERS_PER_WAREHOUSE,
    on_result: Callable[[ResultRow], None] | None = None,
) -> RunSummary:
    """Run tasks with one bounded pool per warehouse; store each row as soon as it's scored.

    Provider errors and gold failures are reported, not stored, so a resumed
    run tries those tasks again. Anything else stops the run.
    """
    adapters = WorkerAdapters(open_adapter)
    data_hashes: dict[tuple[str, Profile], str] = {}
    for task in tasks:
        for profile in PROFILES:
            key = (task.question.schema_name, profile)
            if key not in data_hashes:
                data_hashes[key] = data_sha256(manifest_dir, task.question.schema_name, profile)

    def work(task: Task) -> ResultRow:
        schema_name = task.question.schema_name
        task_adapters: dict[Profile, Warehouse] = {}
        golds: dict[Profile, StoredGold] = {}
        for profile in PROFILES:
            adapter = adapters.get(task.warehouse, schema_name, profile)
            task_adapters[profile] = adapter
            data_hash = data_hashes[(schema_name, profile)]
            golds[profile] = ensure_gold(
                store, task.question, task.warehouse, profile, adapter, data_hash
            )
        client = clients[task.model.id]
        return execute_task(task, run, client, cache, template, golds, task_adapters)

    summary = RunSummary()
    pools: list[ThreadPoolExecutor] = []
    futures: dict[Future[ResultRow], Task] = {}
    try:
        warehouses = sorted({task.warehouse for task in tasks})
        for warehouse in warehouses:
            pool = ThreadPoolExecutor(
                max_workers=workers_per_warehouse, thread_name_prefix=f"run-{warehouse}"
            )
            pools.append(pool)
            for task in tasks:
                if task.warehouse == warehouse:
                    futures[pool.submit(work, task)] = task
        for future in as_completed(futures):
            task = futures[future]
            label = "/".join(task.key)
            try:
                row = future.result()
            except (ProviderError, GoldError) as error:
                summary.failures.append(f"{label}: {error}")
                continue
            store.add_result(row)
            summary.stored += 1
            summary.cost_usd += row.cost_usd
            if on_result is not None:
                on_result(row)
    finally:
        for pool in pools:
            pool.shutdown(wait=True, cancel_futures=True)
        adapters.close_all()
    return summary
