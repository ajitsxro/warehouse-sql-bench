"""Reading configuration from the environment (.env is loaded by the CLI)."""

import os


class MissingEnvError(Exception):
    """Raised when required environment variables are unset or empty."""


def require_env(names: list[str]) -> dict[str, str]:
    """Return the values of names, or raise naming every missing one. Never includes values."""
    missing = [name for name in names if not os.environ.get(name)]
    if missing:
        listed = ", ".join(missing)
        raise MissingEnvError(f"missing environment variables: {listed} (set them in .env)")
    values: dict[str, str] = {}
    for name in names:
        values[name] = os.environ[name]
    return values
