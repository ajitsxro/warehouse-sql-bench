# Changelog

All notable changes to this project are recorded here. After results exist,
every change to a question or its gold SQL is logged with the question id and
the reason, and the affected questions are re-run.

## Unreleased

### Question changes after results (pilot-1, 2026-11-10)

Wording only; gold SQL unchanged. Found when a model's answer was a fair
reading of the question that gold didn't accept. Affected questions are
re-run.

- ecommerce_028, hr_people_018, hr_people_022: "How many ... have 0, 1, 2 ...
  or 4 X?" could mean listing every number, including those with a count of
  0. Gold lists only numbers that occur; the question now says so.
  (claude-opus-5-5 listed zero counts for hr_people_022 on both warehouses.)
- ecommerce_024, hr_people_024: the comma-separated tag/skill strings didn't
  say whether a repeated element is kept. Gold keeps repeats; the question
  now says so. (claude-opus-5-5 removed duplicates for hr_people_024 on both
  warehouses; 21% of employees and 17% of products list an element twice.)

Re-run as pilot-2 (same 40-question sample, same two models). Four of the
five are in the sample; ecommerce_024 isn't, so it has no re-run yet. All
four are correct for both models on both warehouses, except qwen3-coder on
Snowflake for hr_people_022: its FLATTEN drops reviews with no scores, so
the 0 row is missing on edge (a model error).

- Project scaffolding: uv project, `bench` CLI entry point, CI, license.
- Scorer version 1: result normalization, column and row matching, outcomes,
  lucky_match and near-miss diagnostics, as described in docs/scoring.md.
  Rules freeze before the first model run; later changes bump the version
  and require a full re-score.
