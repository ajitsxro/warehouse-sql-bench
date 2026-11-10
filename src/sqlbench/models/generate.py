"""Ask one model for one question's SQL: prompt, cache, call, extract, account."""

from dataclasses import dataclass
from pathlib import Path

from sqlbench.models.cache import ResponseCache, cache_key, sha256_text
from sqlbench.models.clients import Completion, ModelClient
from sqlbench.models.config import ModelConfig
from sqlbench.models.extract import extract_sql
from sqlbench.models.prompt import PromptTemplate, render_prompt
from sqlbench.questions.model import Question
from sqlbench.schema.spec import SchemaSpec
from sqlbench.warehouses.base import WarehouseName

# Rough characters per token, for cost estimates before any call is made.
CHARACTERS_PER_TOKEN = 4


@dataclass(frozen=True)
class Generation:
    question_id: str
    model_id: str
    model_version: str
    warehouse: WarehouseName
    template_sha256: str
    prompt_sha256: str
    completion: Completion
    # None when the response holds no SQL (scored no_sql).
    sql: str | None
    # What this call cost; 0 when the response came from the cache.
    cost_usd: float
    cached: bool


def build_prompt(
    template: PromptTemplate,
    schema: SchemaSpec,
    warehouse: WarehouseName,
    question: Question,
    data_dir: Path,
) -> str:
    return render_prompt(template, schema, warehouse, question.question, data_dir)


def generate(
    question: Question,
    prompt: str,
    warehouse: WarehouseName,
    model: ModelConfig,
    client: ModelClient,
    cache: ResponseCache,
    template: PromptTemplate,
) -> Generation:
    key = cache_key(model, prompt)
    completion = cache.get(key)
    cached = completion is not None
    if completion is None:
        completion = client.complete(prompt)
        cache.put(key, model, prompt, completion)
    cost = 0.0 if cached else model.cost_usd(completion.input_tokens, completion.output_tokens)
    return Generation(
        question_id=question.id,
        model_id=model.id,
        model_version=model.version,
        warehouse=warehouse,
        template_sha256=template.sha256,
        prompt_sha256=sha256_text(prompt),
        completion=completion,
        sql=extract_sql(completion.text, warehouse),
        cost_usd=cost,
        cached=cached,
    )


def estimate_cost(model: ModelConfig, prompts: list[str]) -> float:
    """An upper bound: every prompt at ~4 characters per token, every answer at max_tokens."""
    input_tokens = 0
    for prompt in prompts:
        input_tokens += len(prompt) // CHARACTERS_PER_TOKEN + 1
    output_tokens = model.max_tokens * len(prompts)
    return model.cost_usd(input_tokens, output_tokens)
