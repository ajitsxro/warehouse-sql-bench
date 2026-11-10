# Changelog

All notable changes to this project are recorded here. After results exist,
every change to a question or its gold SQL is logged with the question id and
the reason, and the affected questions are re-run.

## Unreleased

- Project scaffolding: uv project, `bench` CLI entry point, CI, license.
- Scorer version 1: result normalization, column and row matching, outcomes,
  lucky_match and near-miss diagnostics, as described in docs/scoring.md.
  Rules freeze before the first model run; later changes bump the version
  and require a full re-score.
