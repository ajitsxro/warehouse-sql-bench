"""A DuckDB cache of model responses, so a rerun never pays for the same call twice.

The key covers everything that changes a response: model id and version,
the rendered prompt, and the call parameters. Only successful completions
are stored; a failed call is retried on the next run.
"""

import hashlib
import json
from pathlib import Path

import duckdb

from sqlbench.models.clients import Completion
from sqlbench.models.config import ModelConfig

SCHEMA = """
CREATE TABLE IF NOT EXISTS responses (
    cache_key VARCHAR PRIMARY KEY,
    model_id VARCHAR NOT NULL,
    model_version VARCHAR NOT NULL,
    prompt_sha256 VARCHAR NOT NULL,
    params_json VARCHAR NOT NULL,
    response_text VARCHAR NOT NULL,
    input_tokens BIGINT NOT NULL,
    output_tokens BIGINT NOT NULL,
    served_model VARCHAR NOT NULL,
    stop_reason VARCHAR,
    created_at TIMESTAMP NOT NULL DEFAULT current_timestamp
)
"""


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def call_params(model: ModelConfig) -> dict[str, object]:
    """The request settings that can change a response."""
    return {
        "provider": model.provider.value,
        "model": model.model,
        "max_tokens": model.max_tokens,
        "temperature": model.temperature,
    }


def cache_key(model: ModelConfig, prompt: str) -> str:
    payload = {
        "model_id": model.id,
        "model_version": model.version,
        "prompt_sha256": sha256_text(prompt),
        "params": call_params(model),
    }
    return sha256_text(json.dumps(payload, sort_keys=True))


class ResponseCache:
    def __init__(self, path: Path | str) -> None:
        if isinstance(path, Path):
            path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = duckdb.connect(str(path))
        self._connection.execute(SCHEMA)

    def get(self, key: str) -> Completion | None:
        row = self._connection.execute(
            "SELECT response_text, input_tokens, output_tokens, served_model, stop_reason "
            "FROM responses WHERE cache_key = ?",
            [key],
        ).fetchone()
        if row is None:
            return None
        text, input_tokens, output_tokens, served_model, stop_reason = row
        return Completion(text, input_tokens, output_tokens, served_model, stop_reason)

    def put(self, key: str, model: ModelConfig, prompt: str, completion: Completion) -> None:
        # The first stored response for a key stays: reruns read it, never replace it.
        self._connection.execute(
            "INSERT INTO responses (cache_key, model_id, model_version, prompt_sha256, "
            "params_json, response_text, input_tokens, output_tokens, served_model, stop_reason) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT DO NOTHING",
            [
                key,
                model.id,
                model.version,
                sha256_text(prompt),
                json.dumps(call_params(model), sort_keys=True),
                completion.text,
                completion.input_tokens,
                completion.output_tokens,
                completion.served_model,
                completion.stop_reason,
            ],
        )

    def close(self) -> None:
        self._connection.close()
