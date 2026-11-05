# Type mapping

Every column has one logical type in `schemas/<name>.yaml`. The DDL emitters
(`src/sqlbench/schema/ddl.py`) map it to each warehouse, and the generator
writes it to Parquet as shown below.

| Logical | Snowflake | Databricks | Parquet (Arrow) |
|---|---|---|---|
| `int` | `INTEGER` | `INT` | `int32` |
| `bigint` | `BIGINT` | `BIGINT` | `int64` |
| `decimal(p,s)` | `NUMBER(p,s)` | `DECIMAL(p,s)` | `decimal128(p,s)` |
| `double` | `FLOAT` | `DOUBLE` | `float64` |
| `string` | `VARCHAR` | `STRING` | `string` |
| `bool` | `BOOLEAN` | `BOOLEAN` | `bool` |
| `date` | `DATE` | `DATE` | `date32` |
| `timestamp_ntz` | `TIMESTAMP_NTZ` | `TIMESTAMP_NTZ` | `timestamp[us]` |
| `timestamp_tz` | `TIMESTAMP_TZ` | `TIMESTAMP` | `timestamp[us, UTC]` |
| `variant` | `VARIANT` | `VARIANT` | JSON text in `string` |
| `array<T>` | `ARRAY` | `ARRAY<T>` | `list<T>` |
| `struct<...>` | `OBJECT` | `STRUCT<...>` | `struct<...>` |
| `map<K,V>` | `OBJECT` | `MAP<K, V>` | `map<K, V>` |

## Differences that can change results

These are the places where the same logical type does not behave identically.
Some are good material for `semantics` questions; each one must be confirmed
on both warehouses before a question relies on it.

- **Integers.** Snowflake `INTEGER` and `BIGINT` are both `NUMBER(38,0)`.
  Databricks `INT` is 32-bit and `BIGINT` 64-bit, so integer arithmetic that
  overflows on Databricks does not overflow on Snowflake.
- **`timestamp_tz`.** Snowflake `TIMESTAMP_TZ` stores an offset with each value.
  Databricks `TIMESTAMP` stores an instant and displays it in the session time
  zone. The data is written as UTC instants, so both hold the same instants,
  but casting to a date or extracting the hour depends on the session time zone
  on Databricks and on the stored offset on Snowflake.
- **Arrays, structs and maps.** Snowflake uses its semi-structured `ARRAY` and
  `OBJECT`, whose elements are `VARIANT`. Databricks uses typed `ARRAY<T>`,
  `STRUCT` and `MAP`. Values can be compared after normalization, but on Snowflake an
  element needs a cast (`tags[0]::STRING`) to behave like a typed value.
- **Map keys.** Snowflake `OBJECT` keys are always strings. A `map<bigint,V>`
  would come back with string keys on Snowflake. Current schemas only use
  string keys.
- **`variant`.** Parquet has no VARIANT type, so VARIANT columns are written
  as JSON text and parsed during load. JSON `null` is the text `null`; SQL
  NULL is a Parquet null.
- **`double`.** Both are 64-bit IEEE floats. Aggregation order can differ, so
  sums of doubles may differ in the last digits; the scorer has to compare
  doubles with a tolerance.
