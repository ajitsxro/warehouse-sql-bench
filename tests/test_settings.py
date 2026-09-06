import pytest

from sqlbench.settings import MissingEnvError, require_env


def test_require_env_returns_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SQLBENCH_TEST_A", "one")
    monkeypatch.setenv("SQLBENCH_TEST_B", "two")

    assert require_env(["SQLBENCH_TEST_A", "SQLBENCH_TEST_B"]) == {
        "SQLBENCH_TEST_A": "one",
        "SQLBENCH_TEST_B": "two",
    }


def test_require_env_names_every_missing_or_empty_variable_without_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SQLBENCH_TEST_SET", "secret-value")
    monkeypatch.setenv("SQLBENCH_TEST_EMPTY", "")
    monkeypatch.delenv("SQLBENCH_TEST_UNSET", raising=False)

    with pytest.raises(MissingEnvError) as raised:
        require_env(["SQLBENCH_TEST_SET", "SQLBENCH_TEST_EMPTY", "SQLBENCH_TEST_UNSET"])

    message = str(raised.value)
    assert "SQLBENCH_TEST_EMPTY" in message
    assert "SQLBENCH_TEST_UNSET" in message
    assert "SQLBENCH_TEST_SET," not in message
    assert "secret-value" not in message
