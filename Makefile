.PHONY: lint typecheck test all

lint:
	uv run ruff check .
	uv run ruff format --check .

typecheck:
	uv run mypy

test:
	uv run pytest

all: lint typecheck test
