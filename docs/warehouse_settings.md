# Warehouse session settings

Every benchmark session, for loading and for queries, runs with the settings
below. Everything else stays at the platform default, because the defaults
are what a model's SQL meets in practice. NULL ordering in `ORDER BY` is one
example, and it differs between the two warehouses.

## What we set

| Setting | Snowflake | Databricks | DuckDB (tests) |
|---|---|---|---|
| Time zone | `TIMEZONE = 'UTC'`, a session parameter passed at connect | `SET TIME ZONE 'UTC'` run right after connecting | `SET TimeZone = 'UTC'` |
| Statement timeout | `ALTER SESSION SET STATEMENT_TIMEOUT_IN_SECONDS = <timeout>` | `SET STATEMENT_TIMEOUT = <timeout + 10>`, with a client-side cancel at `<timeout>` | client-side interrupt |
| Query tag | `QUERY_TAG = 'sqlbench'` | none | none |
| Current database and schema | `database=` / `schema=` at connect | `catalog=` / `schema=` at connect | `USE "<database>"."<schema>"` |

The code is in `src/sqlbench/warehouses/`. Setting these per session means an
account or user default can't change a result.

## Why the time zone is pinned

The session time zone changes query results, not just how they are displayed:

- A Snowflake `TIMESTAMP_TZ` value keeps the offset it was created with, and
  values loaded from Parquet get the session's offset at load time. A later
  `CAST(ts AS DATE)` uses that stored offset. The account default time zone
  is `America/Los_Angeles`, so data loaded unpinned would put an event at
  02:00 UTC on the previous day.
- A Databricks `TIMESTAMP` is an instant. `CAST(ts AS DATE)`, `date_trunc`
  and `hour()` all use the session time zone when the query runs.

Pinning both warehouses to UTC removes this as a source of differences, so a
time-zone question measures the model's SQL (for example an explicit
`CONVERT_TIMEZONE` / `from_utc_timestamp`), not a configuration accident.

## What we leave alone

Default NULL ordering, `DATEDIFF` boundary counting, integer division,
decimal result precision, case sensitivity of string comparison, and
timestamp type defaults all stay as shipped. Where these differ between the
warehouses, they are what `dialect_sensitive` questions with the `semantics`
tag test, and every such difference is confirmed by running it on both
warehouses before it is used.
