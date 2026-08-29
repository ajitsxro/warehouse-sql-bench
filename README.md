# Warehouse SQL Bench

Warehouse SQL Bench measures how accurately LLMs turn natural-language questions into SQL on Snowflake and on Databricks. Each generated query is run on the warehouse and its result set is compared with the result of a hand-verified gold query. Questions are split between portable SQL and features specific to each warehouse's dialect.

Status: in development.

## License

Code is MIT licensed (see `LICENSE`). The question set will be released under CC BY 4.0.
