"""Runs: resuming by run id, the worker pools, and run fingerprints."""

import threading
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from question_files import question_data, write_question

from sqlbench.datagen.generator import generate_dataset
from sqlbench.datagen.tables import Profile
from sqlbench.harness.executor import Task
from sqlbench.harness.run import (
    RunError,
    build_tasks,
    data_manifest_sha256,
    git_sha,
    pending,
    question_set_sha256,
    run_tasks,
    sample_per_schema,
    start_or_resume,
)
from sqlbench.harness.store import ResultStore, RunInfo
from sqlbench.models.cache import ResponseCache
from sqlbench.models.clients import Completion, ModelClient, ProviderError
from sqlbench.models.config import ModelConfig, Provider
from sqlbench.models.prompt import PromptTemplate, load_template
from sqlbench.questions.model import Question
from sqlbench.schema.spec import load_schema
from sqlbench.warehouses.base import Column, QueryResult, Warehouse, WarehouseName

ROOT = Path(__file__).parent.parent
FIXTURE = Path(__file__).parent / "fixtures" / "mini_shop.yaml"
TEMPLATE = PromptTemplate("{question}", "t" * 64)
RUN = RunInfo("run-1", "abc", "p" * 64, "q" * 64, "d" * 64, 1, ("m1", "m2"), ("snowflake",))


def model(model_id: str) -> ModelConfig:
    return ModelConfig(
        id=model_id,
        provider=Provider.OPENAI,
        model=model_id,
        version=f"{model_id}-v1",
        max_tokens=100,
        temperature=0,
        input_usd_per_million=1.0,
        output_usd_per_million=1.0,
        requests_per_minute=60000,
    )


MODELS = [model("m1"), model("m2")]


def question(number: int) -> Question:
    return Question.model_validate(question_data(id=f"mini_shop_{number:03d}"))


QUESTIONS = [question(1), question(2)]


@pytest.fixture(scope="module")
def manifest_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("run")
    for profile in (Profile.MAIN, Profile.EDGE):
        generate_dataset(FIXTURE, profile, 5, root / "data", root / "manifests")
    return root


class OneRowWarehouse:
    """Returns one row for any SQL; tracks queries, closes and peak concurrency."""

    active = 0
    peak = 0
    lock = threading.Lock()

    def __init__(self, delay_s: float = 0.0) -> None:
        self.name = WarehouseName.SNOWFLAKE
        self.queries: list[str] = []
        self.closed = False
        self.delay_s = delay_s

    def execute(self, sql: str, timeout_s: int) -> QueryResult:
        with OneRowWarehouse.lock:
            OneRowWarehouse.active += 1
            OneRowWarehouse.peak = max(OneRowWarehouse.peak, OneRowWarehouse.active)
        time.sleep(self.delay_s)
        self.queries.append(sql)
        with OneRowWarehouse.lock:
            OneRowWarehouse.active -= 1
        return QueryResult(elapsed_ms=1.0, columns=[Column("c", "FIXED")], rows=[(1,)])

    def close(self) -> None:
        self.closed = True


class Factory:
    def __init__(self, delay_s: float = 0.0) -> None:
        self.opened: list[OneRowWarehouse] = []
        self.delay_s = delay_s
        self.lock = threading.Lock()

    def __call__(self, warehouse: WarehouseName, schema: str, profile: Profile) -> Warehouse:
        adapter = OneRowWarehouse(self.delay_s)
        with self.lock:
            self.opened.append(adapter)
        return adapter

    def all_queries(self) -> list[str]:
        return [sql for adapter in self.opened for sql in adapter.queries]


class FakeModel:
    """Answers SELECT 1; fails with a retryable error for the listed prompts."""

    def __init__(self, failing_prompts: set[str] | None = None) -> None:
        self.failing_prompts = failing_prompts or set()
        self.prompts: list[str] = []
        self.lock = threading.Lock()

    def complete(self, prompt: str) -> Completion:
        with self.lock:
            self.prompts.append(prompt)
        if prompt in self.failing_prompts:
            raise ProviderError("503 overloaded", retryable=True)
        return Completion("```sql\nSELECT 1\n```", 100, 10, "served", "stop")


def tasks(warehouses: tuple[WarehouseName, ...] = (WarehouseName.SNOWFLAKE,)) -> list[Task]:
    built = []
    for warehouse in warehouses:
        for asked in QUESTIONS:
            for each_model in MODELS:
                built.append(Task(each_model, warehouse, asked, f"prompt {asked.id}"))
    return built


def run(
    store: ResultStore,
    task_list: list[Task],
    clients: dict[str, ModelClient],
    factory: Factory,
    manifest_dir: Path,
    workers: int = 2,
) -> Any:
    return run_tasks(
        task_list,
        RUN,
        store,
        ResponseCache(":memory:"),
        TEMPLATE,
        clients,
        factory,
        manifest_dir / "manifests",
        workers_per_warehouse=workers,
    )


def test_a_run_stores_one_row_per_task_and_closes_adapters(manifest_dir: Path) -> None:
    store = ResultStore(":memory:")
    fake = FakeModel()
    factory = Factory()
    seen: list[str] = []

    summary = run_tasks(
        tasks(),
        RUN,
        store,
        ResponseCache(":memory:"),
        TEMPLATE,
        {"m1": fake, "m2": fake},
        factory,
        manifest_dir / "manifests",
        on_result=lambda row: seen.append(row.question_id),
    )

    assert summary.stored == 4
    assert summary.failures == []
    assert summary.cost_usd == pytest.approx(4 * 110 / 1e6)
    assert {row.outcome for row in store.results("run-1")} == {"correct"}
    assert sorted(seen) == ["mini_shop_001"] * 2 + ["mini_shop_002"] * 2
    assert all(adapter.closed for adapter in factory.opened)


def test_resume_runs_only_what_is_missing(manifest_dir: Path) -> None:
    store = ResultStore(":memory:")
    start_or_resume(store, RUN)
    flaky = FakeModel(failing_prompts={"prompt mini_shop_002"})
    clients: dict[str, ModelClient] = {"m1": flaky, "m2": flaky}

    first = run(store, tasks(), clients, Factory(), manifest_dir)

    assert first.stored == 2
    assert len(first.failures) == 2
    assert "m1/snowflake/mini_shop_002: 503 overloaded" in first.failures

    start_or_resume(store, RUN)
    remaining = pending(store, "run-1", tasks())
    healthy = FakeModel()
    second = run(store, remaining, {"m1": healthy, "m2": healthy}, Factory(), manifest_dir)

    assert [task.question.id for task in remaining] == ["mini_shop_002", "mini_shop_002"]
    assert second.stored == 2
    assert healthy.prompts == ["prompt mini_shop_002"] * 2
    assert len(store.results("run-1")) == 4
    assert pending(store, "run-1", tasks()) == []


def test_resume_refuses_a_changed_setup() -> None:
    store = ResultStore(":memory:")
    start_or_resume(store, RUN)

    with pytest.raises(RunError, match="different git sha"):
        start_or_resume(store, replace(RUN, prompt_sha256="x" * 64))
    with pytest.raises(RunError, match="covers models"):
        start_or_resume(store, replace(RUN, models=("m1",)))
    start_or_resume(store, RUN)


def test_each_warehouse_pool_is_bounded(manifest_dir: Path) -> None:
    OneRowWarehouse.active = 0
    OneRowWarehouse.peak = 0
    fake = FakeModel()
    many = [
        replace(task, question=question(number)) for number in range(1, 7) for task in tasks()[:1]
    ]

    summary = run(
        ResultStore(":memory:"), many, {"m1": fake, "m2": fake}, Factory(0.02), manifest_dir
    )

    assert summary.stored == 6
    assert 1 < OneRowWarehouse.peak <= 2


def test_warehouses_get_separate_pools(manifest_dir: Path) -> None:
    fake = FakeModel()
    both = tasks((WarehouseName.SNOWFLAKE, WarehouseName.DATABRICKS))
    store = ResultStore(":memory:")

    summary = run(store, both, {"m1": fake, "m2": fake}, Factory(), manifest_dir)

    assert summary.stored == 8
    assert {row.warehouse for row in store.results("run-1")} == {"snowflake", "databricks"}


def test_gold_runs_once_per_question_and_dataset(manifest_dir: Path) -> None:
    fake = FakeModel()
    factory = Factory()

    run(
        ResultStore(":memory:"), tasks(), {"m1": fake, "m2": fake}, factory, manifest_dir, workers=1
    )

    gold_queries = [sql for sql in factory.all_queries() if "GROUP BY segment" in sql]
    assert len(gold_queries) == 2 * 2


def test_build_tasks_renders_one_prompt_per_warehouse_and_question(manifest_dir: Path) -> None:
    template = load_template(ROOT / "prompts" / "v1.txt")
    schemas = {"mini_shop": load_schema(FIXTURE)}
    warehouses = [WarehouseName.SNOWFLAKE, WarehouseName.DATABRICKS]

    built = build_tasks(MODELS, warehouses, QUESTIONS, schemas, template, manifest_dir / "data")

    assert len(built) == 2 * 2 * 2
    assert built[0].prompt == built[1].prompt
    assert "Snowflake SQL" in built[0].prompt
    assert "Databricks SQL" in built[-1].prompt


def test_question_set_hash_follows_file_contents(tmp_path: Path) -> None:
    first = write_question(tmp_path)
    before = question_set_sha256([first])
    first.write_text(first.read_text().replace("easy", "medium"))

    assert question_set_sha256([first]) != before


def test_data_hash_is_stable(manifest_dir: Path) -> None:
    manifests = manifest_dir / "manifests"

    assert data_manifest_sha256(manifests, ["mini_shop"]) == data_manifest_sha256(
        manifests, ["mini_shop", "mini_shop"]
    )


def test_git_sha_names_the_commit() -> None:
    sha = git_sha(ROOT)

    assert len(sha.removesuffix("-dirty")) == 40


def test_sampling_takes_the_same_mix_from_each_schema() -> None:
    shop = [question(number) for number in range(1, 11)]
    org = [
        Question.model_validate(question_data(id=f"mini_org_{n:03d}", schema="mini_org"))
        for n in range(1, 11)
    ]

    picked = sample_per_schema(shop + org, 3)

    assert sum(1 for q in picked if q.schema_name == "mini_shop") == 3
    assert sum(1 for q in picked if q.schema_name == "mini_org") == 3
    assert picked == sample_per_schema(list(reversed(shop + org)), 3)
    assert [q.id for q in picked] == sorted(q.id for q in picked)
    assert [q.id for q in picked[:3]] != ["mini_org_001", "mini_org_002", "mini_org_003"]
