# Semantics probes

A question gets the `semantics` divergence tag only if a probe here shows
that the same SQL text returns different results on the two warehouses.
Each probe is a small query on literal values (or one loaded table), run as
the read-only benchmark principal with the session settings in
[warehouse_settings.md](warehouse_settings.md).

Run on 2026-11-10 against Snowflake 10.40.103 and Databricks SQL 2026.44.
Behavior can change between releases, so re-run the probes after an
upgrade before trusting a trap.

## Differences that reproduce

### null_order_desc / null_order_asc / null_order_window

```sql
SELECT x FROM (VALUES (1), (NULL), (3)) AS t(x) ORDER BY x DESC
SELECT x FROM (VALUES (1), (NULL), (3)) AS t(x) ORDER BY x
SELECT x, ROW_NUMBER() OVER (ORDER BY x DESC) AS rn
FROM (VALUES (1), (NULL), (3)) AS t(x) ORDER BY rn
```

| Probe | Snowflake | Databricks |
|---|---|---|
| `ORDER BY x DESC` | NULL, 3, 1 | 3, 1, NULL |
| `ORDER BY x` | 1, 3, NULL | NULL, 1, 3 |
| `ROW_NUMBER() OVER (ORDER BY x DESC)` | NULL is 1 | NULL is 3 |

Snowflake treats NULL as larger than any value, and Databricks as smaller.
This applies to window ordering as well.
Used by ecommerce_031, hr_people_031, hr_people_032.

### datediff_day_ts

```sql
SELECT DATEDIFF(day, TIMESTAMP '2024-01-01 23:00:00', TIMESTAMP '2024-01-02 01:00:00') AS d
```

| Snowflake | Databricks |
|---|---|
| 1 | 0 |

Snowflake counts midnights crossed; Databricks counts whole 24-hour
periods. On DATE arguments both return 1 (probe `datediff_day_date`).
Used by ecommerce_032.

### datediff_month / datediff_month_full / datediff_year

```sql
SELECT DATEDIFF(month, DATE '2024-01-31', DATE '2024-02-01') AS d
SELECT DATEDIFF(month, DATE '2024-01-15', DATE '2024-03-14') AS d
SELECT DATEDIFF(year, DATE '2023-12-31', DATE '2024-01-01') AS d
```

| Probe | Snowflake | Databricks |
|---|---|---|
| month, Jan 31 to Feb 1 | 1 | 0 |
| month, Jan 15 to Mar 14 | 2 | 1 |
| year, Dec 31 to Jan 1 | 1 | 0 |

Snowflake counts unit boundaries crossed; Databricks counts completed
units. Used by hr_people_033 and hr_people_034.

### add_months_feb29

```sql
SELECT ADD_MONTHS(DATE '2024-02-29', 1) AS a,
       ADD_MONTHS(DATE '2024-04-30', 1) AS b,
       ADD_MONTHS(DATE '2024-01-31', 1) AS c
```

| Column | Snowflake | Databricks |
|---|---|---|
| a | 2024-03-31 | 2024-03-29 |
| b | 2024-05-31 | 2024-05-30 |
| c | 2024-02-29 | 2024-02-29 |

From the last day of a month, Snowflake lands on the last day of the target
month; Databricks keeps the day number. `DATEADD(month, 1, ...)` keeps the
day number on both (probe `dateadd_month`), though Databricks returns a
TIMESTAMP rather than a DATE. Used by ecommerce_035.

### greatest_null

```sql
SELECT GREATEST(1, NULL) AS g, LEAST(1, NULL) AS l
```

| Snowflake | Databricks |
|---|---|
| NULL, NULL | 1, 1 |

Snowflake returns NULL if any argument is NULL; Databricks skips NULLs.
Used by ecommerce_033 and hr_people_035.

### dayofweek

```sql
SELECT DAYOFWEEK(DATE '2024-01-07') AS sun, DAYOFWEEK(DATE '2024-01-08') AS mon
```

| Snowflake | Databricks |
|---|---|
| 0, 1 | 1, 2 |

Both start the week on Sunday, but Snowflake numbers from 0 (with the
default WEEK_START = 0) and Databricks from 1. Used by ecommerce_034.

### count_variant_jsonnull / count_attributes_edge

```sql
SELECT COUNT(v) AS c, COUNT(*) AS n
FROM (SELECT PARSE_JSON(c) AS v
      FROM (VALUES ('null'), ('{"a":1}'), (NULL)) AS t(c)) AS s

-- on ecommerce__edge
SELECT COUNT(attributes) AS c, COUNT(*) AS n,
       SUM(CASE WHEN attributes IS NULL THEN 1 ELSE 0 END) AS sql_nulls
FROM orders
```

| Probe | Snowflake | Databricks |
|---|---|---|
| literal values: c, n | 1, 3 | 2, 3 |
| ecommerce__edge.orders: c, n, sql_nulls | 569, 600, 30 | 570, 600, 30 |

`COUNT` over a VARIANT skips a JSON null on Snowflake and counts it on
Databricks. `IS NULL` is false for a JSON null on both. Used by
ecommerce_036.

### cast_decimal_int / cast_double_int

```sql
SELECT CAST(1.5 AS INT) AS a, CAST(2.5 AS INT) AS b,
       CAST(-1.5 AS INT) AS c, CAST(1.4 AS INT) AS d
SELECT CAST(CAST(1.7 AS DOUBLE) AS INT) AS a
```

| Probe | Snowflake | Databricks |
|---|---|---|
| decimal to INT | 2, 3, -2, 1 | 1, 2, -1, 1 |
| double 1.7 to INT | 2 | 1 |

Snowflake rounds half away from zero; Databricks truncates toward zero.
Used by hr_people_036.

## Differences in type only

These return the same value with a different type. The scorer's
normalization handles them, so they aren't used as traps:

| Probe | SQL | Snowflake | Databricks |
|---|---|---|---|
| date_minus_date | `SELECT DATE '2024-03-01' - DATE '2024-02-01'` | 29 (integer) | 29 days (interval) |
| dateadd_month | `DATEADD(month, 1, DATE '2024-04-30')` | DATE 2024-05-30 | TIMESTAMP 2024-05-30 00:00 |
| date_trunc_week | `DATE_TRUNC('week', DATE '2024-01-07')` | DATE 2024-01-01 | TIMESTAMP 2024-01-01 00:00 |
| decimal_div | `CAST(10 AS DECIMAL(10,2)) / 3` | 3.33333333 | 3.333333 |
| avg_int | `AVG(x)` over 1, 2 | 1.500000 (NUMBER) | 1.5 (DOUBLE) |
| recursive_cte | `WITH RECURSIVE r (n) AS (...)` | NUMBER | INT |

`date_minus_date` is a `syntax` difference: a question asking for days
between dates needs DATEDIFF to get an integer on both.

## Candidates that did not reproduce (dropped)

| Probe | SQL | Result on both |
|---|---|---|
| datediff_day_date | `DATEDIFF(day, DATE '2024-01-01', DATE '2024-01-02')` | 1 |
| weekofyear | `WEEKOFYEAR` of 2023-01-01, 2024-12-30, 2024-01-01 | 52, 1, 1 |
| date_trunc_week | `DATE_TRUNC('week', ...)` of Sun 2024-01-07, Mon 2024-01-08 | 2024-01-01, 2024-01-08 |
| running_sum_ties | `SUM(v) OVER (ORDER BY k)` with tied k | 30, 30, 35 (RANGE frame on both) |
| int_division | `7 / 2` | 3.5 |
| round_half | `ROUND(2.5)`, `ROUND(-2.5)`, `ROUND(2.5::DOUBLE)` | 3, -3, 3.0 |
| trailing_space_eq | `'a' = 'a '` | false |
| concat_null | `CONCAT('a', NULL)` | NULL |
| string_order | `ORDER BY c` over b, B, a, A | A, B, a, b |
| convert_timezone | `CONVERT_TIMEZONE('UTC', 'America/New_York', TIMESTAMP '2024-03-10 07:00:00')` | 2024-03-10 03:00 |
| dayname | `DAYNAME(DATE '2024-01-08')` | Mon |
| last_day | `LAST_DAY(DATE '2024-02-10')` | 2024-02-29 |
| median | `MEDIAN` over 1, 2, 3, 10 | 2.5 |
| split_part | `SPLIT_PART('a-b-c', '-', 2)` | b |

Portability checks run during drafting (same SQL on both, same result):
`EXTRACT(DOW_ISO FROM ...)`, `attributes:items[0].sku::STRING`,
`specs['color']::STRING`, `attributes:channel::STRING`, `ARRAY_SIZE` on an
ARRAY column, `LISTAGG(DISTINCT ...) WITHIN GROUP (ORDER BY ...)`, and
`WITH RECURSIVE`. These differ in syntax only: `ARRAY_SIZE` on a VARIANT
array (Databricks needs a cast to ARRAY<VARIANT>), `ARRAY_CONTAINS`
argument order, `TO_CHAR` on dates (Databricks rejects the pattern), and
`EXTRACT(DAYOFWEEKISO ...)` (Databricks only accepts DOW_ISO).

## Checking that a trap matters on the data

A probe shows the functions differ, but a question only tests that if the
difference changes its answer. For every `semantics` question, the
natural SQL that ignores the difference was run on the warehouse where it's
wrong and compared with that warehouse's gold:

| Question | Wrong on | main | edge |
|---|---|---|---|
| ecommerce_031 | Snowflake | differs | differs |
| ecommerce_032 | Databricks | differs | differs |
| ecommerce_033 | Snowflake | differs | differs |
| ecommerce_034 | Snowflake | differs | differs |
| ecommerce_035 | Snowflake | differs | differs |
| ecommerce_036 | Databricks | same | differs |
| hr_people_031 | Snowflake | differs | differs |
| hr_people_032 | Databricks | differs | differs |
| hr_people_033 | Databricks | differs | differs |
| hr_people_034 | Snowflake | differs | differs |
| hr_people_035 | Snowflake | differs | differs |
| hr_people_036 | Snowflake | differs | differs |

ecommerce_036 only differs on edge, because the main dataset has no JSON
null payloads. Scoring needs both datasets to match, so the trap still
counts.
