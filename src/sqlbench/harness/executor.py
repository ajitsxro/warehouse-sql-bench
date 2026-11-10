"""One task: a model answers a question for a warehouse, and the answer is run and scored.

The predicted SQL runs on main and on edge (the adapter applies the
read-only guard first) and is scored against gold results computed once and
cached in the store.
"""

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from sqlbench.datagen.generator import file_sha256
from sqlbench.datagen.tables import Profile
from sqlbench.harness.serialize import result_sha256
from sqlbench.harness.store import GoldKey, ResultRow, ResultStore, RunInfo, StoredGold
from sqlbench.models.cache import ResponseCache
from sqlbench.models.clients import ModelClient
from sqlbench.models.config import ModelConfig
from sqlbench.models.generate import generate
from sqlbench.models.prompt import PromptTemplate
from sqlbench.questions.check import GOLD_TIMEOUT_S, PROFILES
from sqlbench.questions.model import Question
from sqlbench.scoring.outcomes import DatasetScore, GoldError, Outcome, combine, score_dataset
from sqlbench.warehouses.base import QueryResult, Warehouse, WarehouseName

PREDICTION_TIMEOUT_S = 60


@dataclass(frozen=True)
class Task:
    model: ModelConfig
    warehouse: WarehouseName
    question: Question
    prompt: str

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.model.id, self.warehouse.value, self.question.id)


def data_sha256(manifest_dir: Path, schema_name: str, profile: Profile) -> str:
    """The committed manifest's hash stands for the dataset's content."""
    return file_sha256(manifest_dir / profile / f"{schema_name}.json")


def gold_key(
    question: Question, warehouse: WarehouseName, profile: Profile, data_sha: str
) -> GoldKey:
    sql = question.gold_sql.for_warehouse(warehouse)
    return GoldKey(
        warehouse=warehouse.value,
        profile=profile.value,
        question_id=question.id,
        gold_sql_sha256=hashlib.sha256(sql.encode()).hexdigest(),
        data_sha256=data_sha,
    )


def ensure_gold(
    store: ResultStore,
    question: Question,
    warehouse: WarehouseName,
    profile: Profile,
    adapter: Warehouse,
    data_sha: str,
) -> StoredGold:
    """The cached gold result, computed and stored on first use."""
    key = gold_key(question, warehouse, profile, data_sha)
    stored = store.get_gold(key)
    if stored is not None:
        return stored
    result = adapter.execute(question.gold_sql.for_warehouse(warehouse), GOLD_TIMEOUT_S)
    if result.error is not None:
        raise GoldError(
            f"{question.id} {warehouse}/{profile}: {result.error.category}: {result.error.message}"
        )
    stored = StoredGold(
        key, question.order_matters, result, result_sha256(result, question.order_matters)
    )
    store.put_gold(stored)
    return stored


def execute_task(
    task: Task,
    run: RunInfo,
    client: ModelClient,
    cache: ResponseCache,
    template: PromptTemplate,
    golds: Mapping[Profile, StoredGold],
    adapters: Mapping[Profile, Warehouse],
) -> ResultRow:
    """Generate, run on both datasets, score, and describe the outcome as one row.

    A ProviderError from the model is not caught: the task isn't stored, so
    a resumed run tries it again.
    """
    question = task.question
    generation = generate(
        question, task.prompt, task.warehouse, task.model, client, cache, template
    )

    scores: dict[Profile, DatasetScore] = {}
    executed: dict[Profile, QueryResult] = {}
    for profile in PROFILES:
        if generation.sql is None:
            scores[profile] = DatasetScore(Outcome.NO_SQL)
            continue
        result = adapters[profile].execute(generation.sql, PREDICTION_TIMEOUT_S)
        executed[profile] = result
        scores[profile] = score_dataset(
            golds[profile].result, result, question.order_matters, question.float_tolerance
        )
    combined = combine(scores[Profile.MAIN], scores[Profile.EDGE])

    near_miss = []
    for profile in PROFILES:
        for label in scores[profile].diagnostics:
            near_miss.append(f"{profile}:{label}")
    error_category = None
    error_message = None
    for profile in PROFILES:
        error = executed[profile].error if profile in executed else None
        if error is not None:
            error_category = error.category.value
            error_message = error.message
            break

    main_result = executed.get(Profile.MAIN)
    edge_result = executed.get(Profile.EDGE)
    return ResultRow(
        run_id=run.run_id,
        git_sha=run.git_sha,
        prompt_sha256=run.prompt_sha256,
        question_set_sha256=run.question_set_sha256,
        data_manifest_sha256=run.data_manifest_sha256,
        scorer_version=combined.scorer_version,
        model_id=task.model.id,
        model_version=task.model.version,
        served_model=generation.completion.served_model,
        warehouse=task.warehouse.value,
        question_id=question.id,
        category=question.category.value,
        divergence=question.divergence.value,
        features=tuple(feature.value for feature in question.features),
        outcome_main=scores[Profile.MAIN].outcome.value,
        outcome_edge=scores[Profile.EDGE].outcome.value,
        outcome=combined.outcome.value,
        lucky_match=combined.lucky_match,
        near_miss=tuple(near_miss),
        predicted_sql=generation.sql,
        error_category=error_category,
        error_message=error_message,
        latency_main_ms=main_result.elapsed_ms if main_result else None,
        latency_edge_ms=edge_result.elapsed_ms if edge_result else None,
        input_tokens=generation.completion.input_tokens,
        output_tokens=generation.completion.output_tokens,
        cost_usd=generation.cost_usd,
        response_cached=generation.cached,
    )
