"""models.yaml: the models under test and what each call costs."""

from enum import StrEnum
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

ID_PATTERN = r"^[a-z0-9][a-z0-9.\-]*$"


class Provider(StrEnum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GOOGLE = "google"
    # OpenAI-compatible APIs: a hosted router for open-weight models, and a local server.
    OPENROUTER = "openrouter"
    OLLAMA = "ollama"


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    id: str = Field(pattern=ID_PATTERN)
    provider: Provider
    # The name sent to the API.
    model: str = Field(min_length=1)
    # The exact version this benchmark means, recorded with every result.
    version: str = Field(min_length=1)
    max_tokens: int = Field(gt=0)
    # None: the parameter is not sent, because the API rejects it.
    temperature: float | None = Field(ge=0)
    input_usd_per_million: float = Field(ge=0)
    output_usd_per_million: float = Field(ge=0)
    requests_per_minute: int = Field(gt=0)

    def cost_usd(self, input_tokens: int, output_tokens: int) -> float:
        input_cost = input_tokens * self.input_usd_per_million / 1_000_000
        output_cost = output_tokens * self.output_usd_per_million / 1_000_000
        return input_cost + output_cost


def load_models(path: Path) -> list[ModelConfig]:
    entries = yaml.safe_load(path.read_text()) or []
    models = [ModelConfig.model_validate(entry) for entry in entries]
    seen: set[str] = set()
    for model in models:
        if model.id in seen:
            raise ValueError(f"{path}: model id {model.id} is listed twice")
        seen.add(model.id)
    return models


def find_model(models: list[ModelConfig], model_id: str) -> ModelConfig:
    for model in models:
        if model.id == model_id:
            return model
    known = ", ".join(model.id for model in models)
    raise KeyError(f"no model {model_id} in models.yaml (known: {known})")
