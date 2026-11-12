# Scoring

A prediction is scored by running it and comparing its result with the
gold result, on the main dataset and on the edge dataset. The SQL text
doesn't matter, only what it returns. The code is in
`src/sqlbench/scoring/`; this page states every rule the code applies.

Scorer version: **1** (`SCORER_VERSION` in `sqlbench/scoring/__init__.py`).
Every stored score records the version that produced it.

## Freezing the rules

The rules on this page are fixed before the first model run. After that, any
change to normalization, matching, outcomes or diagnostics:

1. bumps `SCORER_VERSION`,
2. gets a CHANGELOG entry saying what changed and why,
3. requires re-scoring every stored result with the new version.

Scores from different scorer versions are never mixed in one report.

## 1. Normalizing values

Each value becomes a `(kind, data)` pair (`normalize.py`). Values of
different kinds are never equal, with one exception: all numbers compare
with each other. The column type the warehouse reports is used for one
decision only: whether a string is JSON text.

### Numbers (`normalize_number`, `numbers_equal`)

- `int`, `Decimal` and `float` all compare by value: `3`, `Decimal('3')`
  and `3.0` are equal. Snowflake returns NUMBER as int or Decimal, while
  Databricks returns DOUBLE as float, so the type can't matter.
- If both values are exact (int or Decimal), they are compared exactly,
  so very large integers don't lose digits.
- If either is a float, both are compared as floats. `Decimal('0.1')` equals
  `0.1`.
- **Tolerance.** Each question has a `float_tolerance` t. Two numbers a and b
  are equal when

  ```
  |a - b| <= t                    (absolute)
  or |a - b| <= t * max(|a|, |b|) (relative)
  ```

  Both bounds are inclusive. The absolute bound handles values near zero
  (where a relative bound shrinks to nothing); the relative bound handles
  large values (where an absolute bound like 0.01 is too tight for sums in
  the millions). With t = 0, numbers must be equal exactly (as floats if
  either is a float). The same t is used for every number in the result.
- `-0.0` equals `0.0` and is stored as `0.0`. A Decimal zero with any sign
  or scale is stored as `0`.
- **NaN equals NaN.** In a result comparison, both queries computing NaN in
  the same cell is the same answer. NaN never equals a number, whatever the
  tolerance.
- **Infinity** equals only an infinity of the same sign.
- Booleans are not numbers: `TRUE` doesn't equal `1`.

### Timestamps and dates (`normalize_timestamp`)

- An aware timestamp (Snowflake TIMESTAMP_TZ, Databricks TIMESTAMP) is
  converted to UTC and stored as UTC wall time.
- A naive timestamp (TIMESTAMP_NTZ) is kept as is. Every benchmark session
  runs in UTC (docs/warehouse_settings.md), so a naive value and an aware
  value with the same UTC wall time are equal. This lets a model cast
  between the two timestamp types without being marked wrong for it.
- Comparison is at microsecond precision, Python's datetime resolution.
  Numeric tolerance never applies to timestamps.
- **A date is not equal to the timestamp at its midnight.** A question
  that asks for a date has a date answer; returning `2024-01-01 00:00:00`
  shows the model kept a timestamp it should have converted. If midnight
  timestamps counted as dates, a query that truncated at the wrong place
  (or not at all, on data where times happen to be midnight) would pass.
  Databricks `DATE_TRUNC` returns a TIMESTAMP, so this is a real trap:
  `CAST(DATE_TRUNC('month', ts) AS DATE)` is needed when the question asks
  for a date.
- TIME and INTERVAL values compare by value within their own kind.

### Strings (`normalize_string`)

Exact. No trimming, no case folding, no Unicode normalization. `'web'`,
`'web '` and `'Web'` are three different answers. When a question wants
cleaned text, it says so, and gold cleans it.

### Booleans

`TRUE` equals `TRUE`. Not equal to 1, `'true'` or anything else.

### Semi-structured values (`normalize_semi_structured`)

VARIANT, OBJECT, ARRAY, MAP and STRUCT values become canonical JSON text:

- Snowflake returns them as JSON text, which is parsed. That applies only
  when the column type is VARIANT, OBJECT or ARRAY (or Databricks
  `variant`); JSON-looking text in a string column stays a string.
- Databricks returns arrays as lists, structs as dicts and maps as lists
  of (key, value) pairs, which become JSON objects.
- Keys are sorted, whitespace removed, map keys turned into strings.
- Numbers in JSON are normalized: `1`, `1.0` and `Decimal('1.00')` are the
  same; non-integers become floats. Tolerance doesn't apply inside JSON.
- **Arrays keep their order**: `[1, 2]` differs from `[2, 1]`.
- **Maps and objects compare by key**, whatever order they come in.
- A JSON string is not a plain string. A Snowflake VARIANT holding `"web"`
  (shown with quotes) doesn't equal the VARCHAR `web`. A model that forgets
  `::STRING` returns a visibly different value.
- Dates and timestamps inside a struct become ISO strings.
- Text in a VARIANT column that isn't valid JSON is kept as a plain string.

### NULL

- SQL NULL equals SQL NULL. In a result comparison, both queries leaving a
  cell empty is the same answer.
- SQL NULL is not `''`, not `0`, not `FALSE`, and not JSON null. JSON null is
  a value inside semi-structured data (`Kind.JSON` with data `"null"`);
  SQL NULL is the absence of a value. Both warehouses keep them apart (see
  docs/semantics_probes.md, `count_variant_jsonnull`).

### Other types

Binary values compare by their bytes. A Python type with no rule raises
`TypeError` rather than guessing. It would surface during the gold check,
before any model run.

## 2. Matching columns (`matching.find_column_order`)

- Column names and aliases are ignored.
- The prediction must have exactly as many columns as gold. More or fewer
  is `wrong_columns`.
- With at most 6 columns, the predicted columns may come in any order. For
  each gold column, the candidates are the predicted columns whose values
  alone match it (as a multiset, or in order when order matters). The search
  then tries assignments of candidates and accepts the first one where all
  rows match. Columns that hold the same values (for example two count
  columns) are both candidates, and the row check tells them apart.
- With more than 6 columns, only gold's order is tried. 7! = 5040
  permutations of large results would be too slow, and wide answers usually
  follow the column order the question lists.
- The matched order is stored with a correct score (`column_order`).

## 3. Matching rows (`matching.rows_match`)

- **order_matters: false.** Rows compare as a multiset: order doesn't
  matter, and duplicates count, so `[a, a, b]` doesn't equal `[a, b, b]`.
- **order_matters: true.** Row i of the prediction must equal row i of gold.
- How the multiset match works with tolerance: rows are grouped by their
  non-numeric values, which must match exactly and in equal numbers. Within
  a group only numbers remain. Both sides are sorted by their numbers and
  compared pairwise. Sorting can pair rows badly when several numeric
  columns are each within tolerance, so if the pairwise check fails the
  matcher looks for any pairing (a bipartite matching). That fallback is
  O(n^2) per group and is skipped above 500 rows in one group. Such a group
  is then judged by the sorted pairing alone, which can only err towards
  "not equal".

## 4. Outcomes (`outcomes.score_dataset`)

One outcome per dataset, decided in this order:

| Order | Outcome | When |
|---|---|---|
| 1 | `no_sql` | the model returned no SQL |
| 2 | `syntax_error` | the warehouse rejected the SQL before running (parse or compile error) |
| 2 | `runtime_error` | the query failed while running, or the warehouse refused it (permission, other) |
| 2 | `timeout` | the query ran past the time limit |
| 2 | `blocked` | the read-only guard refused it; it never reached the warehouse |
| 3 | `wrong_columns` | the column count differs from gold |
| 4 | `empty_result` | gold has rows and the prediction has none |
| 5 | `correct` | some allowed column order makes the rows match |
| 5 | `wrong_result` | otherwise |

- Column count is checked before emptiness: a prediction with the wrong
  columns and no rows is `wrong_columns`, the more basic mistake.
- An empty prediction where gold is empty is `correct`. A non-empty
  prediction where gold is empty is `wrong_result`.
- A result with one all-NULL row is not empty. `SELECT SUM(x) FROM t WHERE
  false` returns one row holding NULL, and that differs from no rows.
- Gold that fails raises `GoldError`: a question with broken gold can't
  score anything, and the gold check (`bench questions check`) is supposed
  to catch it first.

### Combining main and edge (`outcomes.combine`)

- The question is `correct` only if both main and edge are `correct`.
- Otherwise, the question's outcome is main's outcome if main failed, and
  edge's outcome if only edge failed.
- `lucky_match` = main correct and edge not correct. These predictions are
  right on realistic data and wrong on the adversarial data: NULLs, ties,
  duplicates, empty groups, boundary dates. They're counted wrong, and the
  flag lets the report show how often that happens.
- Both dataset scores are kept, with the scorer version.

## 5. Near-miss diagnostics (`diagnostics.diagnose`)

For error analysis only. They are computed for `wrong_result` and
`wrong_columns` and **never change an outcome**. Each label re-runs the
comparison with exactly one rule relaxed, so a label means "had only this
been different, it would have matched":

| Label | Relaxed rule |
|---|---|
| `extra_columns_superset` | the prediction has more columns, and some of them (in some order) match gold exactly. Searched when the prediction has at most 12 columns and gold at most 6. |
| `row_subset` | every predicted row is a gold row (counting duplicates); some gold rows are missing |
| `row_superset` | every gold row is predicted (counting duplicates); there are extra rows. Not reported when gold is empty. |
| `rounding_only` | numbers compare equal if they differ by at most half a unit in the last decimal place of the less precise value: 3.5 vs 3.4969 (1 place, 0.05) or 35 vs 35.4 (0 places, 0.5) |
| `order_only` | order_matters is true, and the rows match in another order |
| `null_vs_empty` | SQL NULL is treated as equal to `''`, `[]`, `{}` and JSON null (not to 0) |

Subset and superset compare numbers rounded to 9 significant digits rather
than with the question's tolerance. That's close enough for a diagnostic,
and it lets rows be hashed. Because each label relaxes one rule, a
prediction with two problems at once (for example wrong order and
rounded) gets neither label.

## Gold checks use the same comparison

`bench questions check` scores the Databricks gold result against the
Snowflake gold result with `score_dataset`. A question only becomes
eligible for verification if the two golds are `correct` against each other
on both datasets, under exactly the rules models are scored by.

## Audit (scorer version 1, pilot-1)

Sample: 40 scored predictions from the pilot run (claude-opus-5-5 and
qwen3-coder, 40 questions, both warehouses), stratified 20 the scorer
marked `correct` and 20 it did not, shuffled. The not-correct half was:
11 syntax_error, 6 wrong_result, 1 wrong_columns, 1 empty_result,
1 runtime_error. The five questions reworded after the pilot (CHANGELOG)
were left out. Each item showed the question, both SQL texts and both
results on main and edge.

Who labelled: a separate Claude agent with no access to the scorer's
verdicts, not a person. This is weaker evidence than a hand audit: the
labeller is an LLM, and Claude is also one of the models being scored.

Result: 40 of 40 labels agree with the scorer (20 of 20 correct,
20 of 20 not), so there were no disagreements to investigate. Three
items the labeller flagged as close calls, all decided the same way as
the scorer:

- An extra column not asked for (`wrong_columns`): judged wrong.
- A midnight timestamp where the question asks for a date: judged wrong,
  which matches the rule that a date never equals a timestamp.
- A `NULLIF(channel, '')` the gold doesn't have: judged correct: it
  changes nothing on these datasets, and the results match on both.

## Known limits

- The bipartite fallback is skipped for groups above 500 rows (section 3).
- Tolerance doesn't apply inside JSON values.
- Wide results (more than 6 columns) must use gold's column order.
- Diagnostics of results with more than 6 columns only consider gold's
  column order.
