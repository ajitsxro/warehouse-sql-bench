# Warehouse SQL Bench

Warehouse SQL Bench measures how accurately LLMs turn natural-language questions into SQL on Snowflake and on Databricks.

The same questions run on identical data in both warehouses, so the comparison isolates the SQL dialect. Each prediction is scored by its result set against gold SQL, on a realistic dataset and on a small edge-case dataset.

Status: in development.

## License

Code is MIT licensed (see `LICENSE`). The question set will be released under CC BY 4.0.
