"""Thin clients for each provider, plus one retry and rate-limit policy for all of them.

Each client sends the prompt as a single user message and returns the text
and the token counts the provider bills. The SDKs' own retries are turned
off so every provider retries the same way (call_with_retries).
"""

import os
import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import anthropic
import openai
from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types

from sqlbench.models.config import ModelConfig, Provider
from sqlbench.settings import require_env

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OLLAMA_DEFAULT_BASE_URL = "http://localhost:11434/v1"
REQUEST_TIMEOUT_S = 600.0
# Rate limits, request timeouts, conflicts and server errors are worth retrying.
RETRYABLE_STATUS_CODES = frozenset({408, 409, 429})


@dataclass(frozen=True)
class Completion:
    text: str
    input_tokens: int
    # Includes reasoning or thinking tokens, which providers bill as output.
    output_tokens: int
    # The model the provider says answered; for OpenRouter, also the upstream provider.
    served_model: str
    stop_reason: str | None


class ProviderError(Exception):
    """A failed call. retryable is True for rate limits, timeouts and server errors."""

    def __init__(self, message: str, retryable: bool) -> None:
        super().__init__(message)
        self.retryable = retryable


class ModelClient(Protocol):
    def complete(self, prompt: str) -> Completion: ...


def is_retryable_status(status_code: int) -> bool:
    return status_code in RETRYABLE_STATUS_CODES or status_code >= 500


class AnthropicClient:
    def __init__(self, config: ModelConfig, sdk: Any | None = None) -> None:
        self._config = config
        if sdk is None:
            api_key = require_env(["ANTHROPIC_API_KEY"])["ANTHROPIC_API_KEY"]
            sdk = anthropic.Anthropic(api_key=api_key, max_retries=0, timeout=REQUEST_TIMEOUT_S)
        self._sdk = sdk

    def complete(self, prompt: str) -> Completion:
        options: dict[str, Any] = {}
        if self._config.temperature is not None:
            options["temperature"] = self._config.temperature
        try:
            response = self._sdk.messages.create(
                model=self._config.model,
                max_tokens=self._config.max_tokens,
                messages=[{"role": "user", "content": prompt}],
                **options,
            )
        except anthropic.APIStatusError as error:
            raise ProviderError(str(error), is_retryable_status(error.status_code)) from error
        except anthropic.APIConnectionError as error:
            raise ProviderError(str(error), retryable=True) from error
        texts = [block.text for block in response.content if block.type == "text"]
        return Completion(
            text="".join(texts),
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            served_model=response.model,
            stop_reason=response.stop_reason,
        )


class OpenAICompatibleClient:
    """OpenAI itself, OpenRouter and Ollama all speak the Chat Completions API."""

    def __init__(self, config: ModelConfig, sdk: Any | None = None) -> None:
        self._config = config
        if sdk is None:
            sdk = openai.OpenAI(**openai_connection(config.provider))
        self._sdk = sdk

    def complete(self, prompt: str) -> Completion:
        options: dict[str, Any] = {}
        if self._config.temperature is not None:
            options["temperature"] = self._config.temperature
        # OpenAI's reasoning models only accept max_completion_tokens.
        if self._config.provider == Provider.OPENAI:
            options["max_completion_tokens"] = self._config.max_tokens
        else:
            options["max_tokens"] = self._config.max_tokens
        try:
            response = self._sdk.chat.completions.create(
                model=self._config.model,
                messages=[{"role": "user", "content": prompt}],
                **options,
            )
        except openai.APIStatusError as error:
            raise ProviderError(str(error), is_retryable_status(error.status_code)) from error
        except openai.APIConnectionError as error:
            raise ProviderError(str(error), retryable=True) from error
        choice = response.choices[0]
        served_model = response.model
        # OpenRouter adds the upstream provider that served the request.
        upstream = getattr(response, "provider", None)
        if upstream:
            served_model = f"{served_model} via {upstream}"
        usage = response.usage
        return Completion(
            text=choice.message.content or "",
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            served_model=served_model,
            stop_reason=choice.finish_reason,
        )


def openai_connection(provider: Provider) -> dict[str, Any]:
    connection: dict[str, Any] = {"max_retries": 0, "timeout": REQUEST_TIMEOUT_S}
    if provider == Provider.OPENAI:
        connection["api_key"] = require_env(["OPENAI_API_KEY"])["OPENAI_API_KEY"]
    elif provider == Provider.OPENROUTER:
        connection["api_key"] = require_env(["OPENROUTER_API_KEY"])["OPENROUTER_API_KEY"]
        connection["base_url"] = OPENROUTER_BASE_URL
    else:
        # Ollama needs no key, but the SDK requires a non-empty one.
        connection["api_key"] = "ollama"
        connection["base_url"] = os.environ.get("OLLAMA_BASE_URL") or OLLAMA_DEFAULT_BASE_URL
    return connection


class GeminiClient:
    def __init__(self, config: ModelConfig, sdk: Any | None = None) -> None:
        self._config = config
        if sdk is None:
            api_key = require_env(["GOOGLE_API_KEY"])["GOOGLE_API_KEY"]
            sdk = genai.Client(api_key=api_key)
        self._sdk = sdk

    def complete(self, prompt: str) -> Completion:
        settings = genai_types.GenerateContentConfig(
            temperature=self._config.temperature,
            max_output_tokens=self._config.max_tokens,
        )
        try:
            response = self._sdk.models.generate_content(
                model=self._config.model, contents=prompt, config=settings
            )
        except genai_errors.APIError as error:
            raise ProviderError(str(error), is_retryable_status(error.code)) from error
        usage = response.usage_metadata
        input_tokens = (usage.prompt_token_count or 0) if usage else 0
        output_tokens = 0
        if usage:
            # Thinking tokens are billed as output on top of the answer.
            output_tokens = (usage.candidates_token_count or 0) + (usage.thoughts_token_count or 0)
        stop_reason = None
        if response.candidates and response.candidates[0].finish_reason:
            stop_reason = str(response.candidates[0].finish_reason)
        return Completion(
            text=response.text or "",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            served_model=response.model_version or self._config.model,
            stop_reason=stop_reason,
        )


def make_client(config: ModelConfig) -> ModelClient:
    if config.provider == Provider.ANTHROPIC:
        return AnthropicClient(config)
    if config.provider == Provider.GOOGLE:
        return GeminiClient(config)
    return OpenAICompatibleClient(config)


class RateLimiter:
    """Spaces calls at least 60 / requests_per_minute seconds apart."""

    def __init__(
        self,
        requests_per_minute: int,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._interval = 60.0 / requests_per_minute
        self._clock = clock
        self._sleep = sleep
        self._last_call: float | None = None

    def wait(self) -> None:
        if self._last_call is not None:
            remaining = self._last_call + self._interval - self._clock()
            if remaining > 0:
                self._sleep(remaining)
        self._last_call = self._clock()


@dataclass(frozen=True)
class RetryPolicy:
    attempts: int = 5
    base_delay_s: float = 2.0
    max_delay_s: float = 60.0


DEFAULT_RETRY_POLICY = RetryPolicy()


def call_with_retries(
    call: Callable[[], Completion],
    policy: RetryPolicy,
    sleep: Callable[[float], None] = time.sleep,
    jitter: Callable[[], float] = random.random,
) -> Completion:
    """Retry retryable ProviderErrors with exponential backoff; raise the rest at once."""
    for attempt in range(policy.attempts):
        try:
            return call()
        except ProviderError as error:
            if not error.retryable or attempt == policy.attempts - 1:
                raise
            delay = min(policy.base_delay_s * 2**attempt, policy.max_delay_s)
            # Up to a second of jitter so parallel runs don't retry in lockstep.
            sleep(delay + jitter())
    raise AssertionError("unreachable: the last attempt returns or raises")


class PacedClient:
    """A client that waits for the rate limiter and retries per the policy."""

    def __init__(
        self,
        client: ModelClient,
        limiter: RateLimiter,
        policy: RetryPolicy = DEFAULT_RETRY_POLICY,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._client = client
        self._limiter = limiter
        self._policy = policy
        self._sleep = sleep

    def complete(self, prompt: str) -> Completion:
        def attempt() -> Completion:
            self._limiter.wait()
            return self._client.complete(prompt)

        return call_with_retries(attempt, self._policy, self._sleep)
