"""The response cache: what the key covers, and what is stored."""

from pathlib import Path
from typing import Any

import pytest

from sqlbench.models.cache import ResponseCache, cache_key
from sqlbench.models.clients import Completion
from sqlbench.models.config import ModelConfig, Provider


def model(**changes: Any) -> ModelConfig:
    fields: dict[str, Any] = {
        "id": "fake-1",
        "provider": Provider.OPENAI,
        "model": "fake-model",
        "version": "fake-model-2026",
        "max_tokens": 1000,
        "temperature": 0.0,
        "input_usd_per_million": 1.0,
        "output_usd_per_million": 2.0,
        "requests_per_minute": 60,
    }
    fields.update(changes)
    return ModelConfig(**fields)


COMPLETION = Completion("```sql\nSELECT 1\n```", 1200, 80, "fake-model-served", "stop")


def test_the_key_is_stable() -> None:
    assert cache_key(model(), "prompt") == cache_key(model(), "prompt")


@pytest.mark.parametrize(
    "changes",
    [
        {"id": "fake-2"},
        {"version": "fake-model-2027"},
        {"model": "other-model"},
        {"max_tokens": 2000},
        {"temperature": None},
        {"provider": Provider.OPENROUTER},
    ],
)
def test_the_key_changes_with_anything_that_changes_a_response(changes: dict[str, Any]) -> None:
    assert cache_key(model(**changes), "prompt") != cache_key(model(), "prompt")


def test_the_key_changes_with_the_prompt() -> None:
    assert cache_key(model(), "prompt A") != cache_key(model(), "prompt B")


def test_prices_and_rate_limits_are_not_part_of_the_key() -> None:
    cheaper = model(input_usd_per_million=0.5, requests_per_minute=5)

    assert cache_key(cheaper, "prompt") == cache_key(model(), "prompt")


def test_round_trip() -> None:
    cache = ResponseCache(":memory:")
    key = cache_key(model(), "prompt")

    assert cache.get(key) is None
    cache.put(key, model(), "prompt", COMPLETION)
    assert cache.get(key) == COMPLETION


def test_the_first_response_is_kept() -> None:
    cache = ResponseCache(":memory:")
    key = cache_key(model(), "prompt")

    cache.put(key, model(), "prompt", COMPLETION)
    cache.put(key, model(), "prompt", Completion("other", 1, 1, "x", None))

    assert cache.get(key) == COMPLETION


def test_responses_persist_in_the_file(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "cache.duckdb"
    key = cache_key(model(), "prompt")
    first = ResponseCache(path)
    first.put(key, model(), "prompt", COMPLETION)
    first.close()

    second = ResponseCache(path)

    assert second.get(key) == COMPLETION
    second.close()


def test_a_missing_stop_reason_and_empty_text_are_stored() -> None:
    cache = ResponseCache(":memory:")
    key = cache_key(model(), "p")
    empty = Completion("", 10, 0, "m", None)

    cache.put(key, model(), "p", empty)

    assert cache.get(key) == empty
