# Dataset card: Warehouse SQL Bench v1.0

## Summary

| | |
|---|---|
| Questions | 72 verified, frozen in `questions/MANIFEST.json` (tag `v1.0.1-inputs`) |
| Schemas with questions | ecommerce, hr_people (36 each) |
| Datasets per schema | main (realistic) and edge (small, adversarial) |
| Gold SQL | one per warehouse: Snowflake and Databricks |
| Scoring | execution; correct only if the result matches gold on main and edge |
| License | questions and gold SQL: CC BY 4.0; code (including the data generator): MIT |

## What a question looks like

Each question is one YAML file in `questions/<schema>/`:

- `question`: the natural-language question, saying how to treat
  anything ambiguous (time zone, how to trim or case strings, ties,
  whether empty groups appear).
- `gold_sql.snowflake` and `gold_sql.databricks`.
- `category`: `portable` (one SQL text works on both warehouses) or
  `dialect_sensitive`.
- `divergence`: `none`, `syntax` (same capability, different spelling,
  e.g. LATERAL FLATTEN vs explode) or `semantics` (SQL that looks portable
  but returns different results by default). Every semantics question
  names the probe that confirmed its trap (`docs/semantics_probes.md`).
- `features` (used only to slice results), `difficulty`,
  `order_matters`, `float_tolerance`.
- `status`, `verified_by`, `verified_at`.

## Splits

There is no train/test split. All 72 questions are evaluation questions.

| | ecommerce | hr_people |
|---|---|---|
| portable | 18 | 18 |
| dialect_sensitive, syntax | 12 | 12 |
| dialect_sensitive, semantics | 6 | 6 |

Difficulty: 21 easy, 34 medium, 17 hard.

## How the questions were written and verified

1. Drafting. Questions and both golds were drafted with an AI coding
   assistant (Claude), schema by schema. Each covers features the schema
   was designed for: JSON paths, arrays, maps, hierarchies, history tables
   and decimals.
2. Semantics probes. Each candidate semantics trap was first run as a
   small probe on both warehouses (`docs/semantics_probes.md`). Traps that
   didn't reproduce were dropped.
3. Lint (`bench questions lint`). Format, schema references in each
   dialect, the read-only guard, bare table names, deterministic LIMIT,
   and no hyphen split by YAML line folding.
4. Gold check (`bench questions check`). The Snowflake and Databricks
   golds must give the same normalized result, scored by the benchmark's
   own scorer, on both main and edge.
5. Review. The maintainer reviewed the questions in
   `bench questions verify`. All 72 were then stamped verified in one batch
   on 2026-11-10, rather than one keypress per question.
6. Changes after results. These are allowed only to fix a verified
   error, and are logged in `CHANGELOG.md`. Eight questions were reworded
   with gold SQL unchanged: five after the pilot (ambiguous zero counts and
   duplicate elements) and three after the first full run (one ambiguity,
   three line-folding typos). The affected questions were re-run each time.

## Data

The data is synthetic, generated deterministically from `schemas/*.yaml`
with a fixed seed (20261105). Each column has its own seeded generator.
`manifests/` records a SHA-256 for every generated file, and CI regenerates
the edge data and checks those hashes. The same Parquet files are loaded
into both warehouses, with sessions pinned to UTC.

| schema | main rows | edge rows |
|---|---|---|
| ecommerce | 770,000 | 2,700 |
| hr_people | 800,012 | 3,824 |

Four more schemas (saas_subscriptions, web_analytics, iot_telemetry,
finance_ledger) are defined and generated but have no questions in v1.0.

### The edge dataset

The edge dataset is generated small and then changed by seven injectors,
in this order (`src/sqlbench/datagen/edge.py`):

| injector | what it guarantees |
|---|---|
| empty_groups | some parent rows (e.g. customers) have no children |
| nulls | every nullable column, foreign keys included, is at least 5% NULL |
| boundary_dates | month ends, Feb 29, the year change, midnight, DST jumps |
| semi_structured | missing JSON keys, JSON null vs SQL NULL, empty and single-element arrays, empty maps |
| messy_strings | leading and trailing spaces, upper and mixed case |
| ties | equal values at rank 1-2 and at the rank 10-11 cut |
| duplicates | exact duplicate rows in tables without a primary key |

The injectors never change primary keys, and they keep foreign keys
valid. Their purpose is SQL that is only accidentally right on realistic
data. In run v1.0.1-full the edge dataset caught 8 such answers: right on
main and wrong on edge.

## Known limits

- The data is synthetic. Some cross-table timing isn't modelled (for
  example, orders against signup dates), and the schema descriptions list
  what questions must not depend on.
- Main data for hr_people has 2024 hires only from January to June.
- 72 questions is too few to separate models near 100% accuracy.

## License

The questions and gold SQL in `questions/` are licensed CC BY 4.0
(`questions/LICENSE`). The generator and the rest of the code are MIT.
Because the data is generated deterministically by that code, it is
distributed by regenerating it, not as files.
