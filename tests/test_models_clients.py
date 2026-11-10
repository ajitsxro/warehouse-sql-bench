"""Provider clients against fake SDK objects, and the shared retry and rate-limit policy."""

from types import SimpleNamespace
from typing import Any

import anthropic
import httpx2
import openai
import pytest
from google.genai import errors as genai_errors

from sqlbench.models.clients import (
    AnthropicClient,
    Completion,
    GeminiClient,
    OpenAICompatibleClient,
    PacedClient,
    ProviderError,
    RateLimiter,
    RetryPolicy,
    call_with_retries,
    make_client,
)
from sqlbench.models.config import ModelConfig, Provider
from sqlbench.settings import MissingEnvError

REQUEST = httpx2.Request("POST", "https://api.example.com/v1")


def config(provider: Provider, temperature: float | None = 0.0) -> ModelConfig:
    return ModelConfig(
        id="m",
        provider=provider,
        model="model-x",
        version="model-x-1",
        max_tokens=500,
        temperature=temperature,
        input_usd_per_million=1.0,
        output_usd_per_million=2.0,
        requests_per_minute=60,
    )


class Recorder:
    """Stands in for an SDK method: records keyword arguments, returns or raises."""

    def __init__(self, result: Any = None, error: Exception | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.result = result
        self.error = error

    def __call__(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.result


def status_error(module: Any, status: int) -> Exception:
    response = httpx2.Response(status, request=REQUEST)
    error: Exception = module.APIStatusError("failed", response=response, body=None)
    return error


# ---------------- Anthropic ----------------


def anthropic_sdk(recorder: Recorder) -> Any:
    return SimpleNamespace(messages=SimpleNamespace(create=recorder))


ANTHROPIC_RESPONSE = SimpleNamespace(
    content=[
        SimpleNamespace(type="thinking", thinking=""),
        SimpleNamespace(type="text", text="```sql\nSELECT 1\n```"),
        SimpleNamespace(type="text", text=" done"),
    ],
    usage=SimpleNamespace(input_tokens=1200, output_tokens=340),
    model="claude-opus-5-5",
    stop_reason="end_turn",
)


def test_anthropic_joins_text_blocks_and_reports_usage() -> None:
    recorder = Recorder(ANTHROPIC_RESPONSE)
    client = AnthropicClient(config(Provider.ANTHROPIC, temperature=None), anthropic_sdk(recorder))

    completion = client.complete("prompt")

    assert completion == Completion(
        "```sql\nSELECT 1\n``` done", 1200, 340, "claude-opus-5-5", "end_turn"
    )
    assert recorder.calls == [
        {
            "model": "model-x",
            "max_tokens": 500,
            "messages": [{"role": "user", "content": "prompt"}],
        }
    ]


def test_anthropic_sends_temperature_only_when_configured() -> None:
    recorder = Recorder(ANTHROPIC_RESPONSE)
    AnthropicClient(config(Provider.ANTHROPIC, 0.0), anthropic_sdk(recorder)).complete("p")

    assert recorder.calls[0]["temperature"] == 0.0


@pytest.mark.parametrize(
    ("status", "retryable"), [(429, True), (529, True), (400, False), (401, False)]
)
def test_anthropic_status_errors_are_classified(status: int, retryable: bool) -> None:
    recorder = Recorder(error=status_error(anthropic, status))
    client = AnthropicClient(config(Provider.ANTHROPIC), anthropic_sdk(recorder))

    with pytest.raises(ProviderError) as raised:
        client.complete("p")

    assert raised.value.retryable is retryable


def test_anthropic_connection_errors_are_retryable() -> None:
    recorder = Recorder(error=anthropic.APIConnectionError(request=REQUEST))
    client = AnthropicClient(config(Provider.ANTHROPIC), anthropic_sdk(recorder))

    with pytest.raises(ProviderError) as raised:
        client.complete("p")

    assert raised.value.retryable


# ---------------- OpenAI-compatible ----------------


def openai_sdk(recorder: Recorder) -> Any:
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=recorder)))


def chat_response(content: str | None, usage: Any, provider: str | None = None) -> Any:
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content), finish_reason="stop")],
        usage=usage,
        model="served-model",
    )
    if provider is not None:
        response.provider = provider
    return response


def test_openai_uses_max_completion_tokens() -> None:
    usage = SimpleNamespace(prompt_tokens=900, completion_tokens=120)
    recorder = Recorder(chat_response("SELECT 1", usage))
    client = OpenAICompatibleClient(config(Provider.OPENAI), openai_sdk(recorder))

    completion = client.complete("prompt")

    assert completion == Completion("SELECT 1", 900, 120, "served-model", "stop")
    assert recorder.calls[0]["max_completion_tokens"] == 500
    assert "max_tokens" not in recorder.calls[0]
    assert recorder.calls[0]["temperature"] == 0.0


def test_openrouter_records_the_upstream_provider() -> None:
    usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5)
    recorder = Recorder(chat_response("x", usage, provider="Fireworks"))
    client = OpenAICompatibleClient(config(Provider.OPENROUTER), openai_sdk(recorder))

    completion = client.complete("p")

    assert completion.served_model == "served-model via Fireworks"
    assert recorder.calls[0]["max_tokens"] == 500


def test_missing_content_and_usage_become_empty_and_zero() -> None:
    recorder = Recorder(chat_response(None, None))
    client = OpenAICompatibleClient(config(Provider.OLLAMA, None), openai_sdk(recorder))

    completion = client.complete("p")

    assert (completion.text, completion.input_tokens, completion.output_tokens) == ("", 0, 0)
    assert "temperature" not in recorder.calls[0]


@pytest.mark.parametrize(("status", "retryable"), [(429, True), (503, True), (404, False)])
def test_openai_status_errors_are_classified(status: int, retryable: bool) -> None:
    recorder = Recorder(error=status_error(openai, status))
    client = OpenAICompatibleClient(config(Provider.OPENAI), openai_sdk(recorder))

    with pytest.raises(ProviderError) as raised:
        client.complete("p")

    assert raised.value.retryable is retryable


# ---------------- Gemini ----------------


def gemini_sdk(recorder: Recorder) -> Any:
    return SimpleNamespace(models=SimpleNamespace(generate_content=recorder))


def test_gemini_counts_thinking_tokens_as_output() -> None:
    response = SimpleNamespace(
        text="SELECT 1",
        usage_metadata=SimpleNamespace(
            prompt_token_count=800, candidates_token_count=50, thoughts_token_count=700
        ),
        candidates=[SimpleNamespace(finish_reason="STOP")],
        model_version="gemini-4-carbon-001",
    )
    recorder = Recorder(response)
    client = GeminiClient(config(Provider.GOOGLE), gemini_sdk(recorder))

    completion = client.complete("prompt")

    assert completion == Completion("SELECT 1", 800, 750, "gemini-4-carbon-001", "STOP")
    settings = recorder.calls[0]["config"]
    assert (settings.temperature, settings.max_output_tokens) == (0.0, 500)
    assert recorder.calls[0]["contents"] == "prompt"


def test_gemini_blocked_response_has_empty_text() -> None:
    response = SimpleNamespace(text=None, usage_metadata=None, candidates=[], model_version=None)
    client = GeminiClient(config(Provider.GOOGLE), gemini_sdk(Recorder(response)))

    assert client.complete("p") == Completion("", 0, 0, "model-x", None)


@pytest.mark.parametrize(("status", "retryable"), [(429, True), (500, True), (400, False)])
def test_gemini_api_errors_are_classified(status: int, retryable: bool) -> None:
    error = genai_errors.APIError(status, {"error": {"message": "failed", "status": "X"}})
    client = GeminiClient(config(Provider.GOOGLE), gemini_sdk(Recorder(error=error)))

    with pytest.raises(ProviderError) as raised:
        client.complete("p")

    assert raised.value.retryable is retryable


# ---------------- construction ----------------


@pytest.mark.parametrize(
    ("provider", "variable"),
    [
        (Provider.ANTHROPIC, "ANTHROPIC_API_KEY"),
        (Provider.OPENAI, "OPENAI_API_KEY"),
        (Provider.GOOGLE, "GOOGLE_API_KEY"),
        (Provider.OPENROUTER, "OPENROUTER_API_KEY"),
    ],
)
def test_clients_need_their_api_key(
    provider: Provider, variable: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(variable, raising=False)

    with pytest.raises(MissingEnvError, match=variable):
        make_client(config(provider))


def test_ollama_needs_no_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)

    assert isinstance(make_client(config(Provider.OLLAMA)), OpenAICompatibleClient)


# ---------------- retries and rate limits ----------------


class FlakyClient:
    def __init__(self, failures: list[ProviderError]) -> None:
        self.failures = failures
        self.calls = 0

    def complete(self, prompt: str) -> Completion:
        self.calls += 1
        if self.failures:
            raise self.failures.pop(0)
        return Completion("ok", 1, 1, "m", "stop")


def test_retryable_errors_back_off_exponentially() -> None:
    client = FlakyClient([ProviderError("429", True), ProviderError("503", True)])
    sleeps: list[float] = []

    completion = call_with_retries(
        lambda: client.complete("p"), RetryPolicy(attempts=3), sleeps.append, lambda: 0.5
    )

    assert completion.text == "ok"
    assert sleeps == [2.5, 4.5]


def test_backoff_is_capped() -> None:
    client = FlakyClient([ProviderError("429", True)] * 3)
    sleeps: list[float] = []
    policy = RetryPolicy(attempts=4, base_delay_s=10.0, max_delay_s=15.0)

    call_with_retries(lambda: client.complete("p"), policy, sleeps.append, lambda: 0.0)

    assert sleeps == [10.0, 15.0, 15.0]


def test_retries_give_up_after_the_last_attempt() -> None:
    client = FlakyClient([ProviderError("429", True)] * 5)

    with pytest.raises(ProviderError, match="429"):
        call_with_retries(lambda: client.complete("p"), RetryPolicy(attempts=2), lambda _: None)

    assert client.calls == 2


def test_non_retryable_errors_raise_at_once() -> None:
    client = FlakyClient([ProviderError("400", False)])

    with pytest.raises(ProviderError, match="400"):
        call_with_retries(lambda: client.complete("p"), RetryPolicy(), lambda _: None)

    assert client.calls == 1


class FakeClock:
    def __init__(self) -> None:
        self.now = 100.0
        self.sleeps: list[float] = []

    def time(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def test_rate_limiter_spaces_calls() -> None:
    clock = FakeClock()
    limiter = RateLimiter(30, clock.time, clock.sleep)

    limiter.wait()
    clock.now += 0.5
    limiter.wait()
    clock.now += 5.0
    limiter.wait()

    assert clock.sleeps == [1.5]


def test_paced_client_waits_before_every_attempt() -> None:
    clock = FakeClock()
    flaky = FlakyClient([ProviderError("429", True)])
    paced = PacedClient(
        flaky, RateLimiter(60, clock.time, clock.sleep), RetryPolicy(base_delay_s=0.0), clock.sleep
    )

    assert paced.complete("p").text == "ok"
    assert flaky.calls == 2
