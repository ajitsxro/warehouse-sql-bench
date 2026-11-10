"""Pull the SQL out of a model's response.

The rule: the first ```sql block; otherwise the whole response, if it parses
as SQL in the warehouse's dialect; otherwise there is no SQL (scored
no_sql). Reasoning inside <think> tags is dropped first, because reasoning
models draft SQL there before their answer.
"""

import re

import sqlglot
from sqlglot import exp
from sqlglot.errors import SqlglotError

from sqlbench.warehouses.base import WarehouseName

# ```sql or ```SQL, possibly with more on the fence line; a block that never
# closes (the response was cut off) runs to the end of the text.
SQL_BLOCK = re.compile(r"```[ \t]*sql\b[^\n]*\n(.*?)(?:```|\Z)", re.IGNORECASE | re.DOTALL)
THINK_BLOCK = re.compile(r"<think>.*?(?:</think>|\Z)", re.IGNORECASE | re.DOTALL)
# Statement types a bare response may consist of. Writes are kept so the
# read-only guard can block them, which scores differently from no_sql.
STATEMENT_TYPES = (exp.Query, exp.DML, exp.DDL, exp.Drop, exp.Alter)


def extract_sql(response: str, dialect: WarehouseName) -> str | None:
    text = THINK_BLOCK.sub("", response)
    match = SQL_BLOCK.search(text)
    if match is not None:
        candidate = match.group(1)
    elif parses_as_sql(text, dialect):
        candidate = text
    else:
        return None
    cleaned = strip_terminators(candidate)
    return cleaned or None


def strip_terminators(sql: str) -> str:
    """Remove surrounding whitespace and any trailing semicolons."""
    cleaned = sql.strip()
    while cleaned.endswith(";"):
        cleaned = cleaned[:-1].rstrip()
    return cleaned


def parses_as_sql(text: str, dialect: WarehouseName) -> bool:
    """Whether every statement in text parses as a real SQL statement.

    Prose either fails to parse or parses as something that isn't a
    statement (sqlglot reads "Hello" as a column), so both are rejected.
    """
    if not text.strip():
        return False
    try:
        statements = sqlglot.parse(text, read=dialect.value)
    except SqlglotError:
        return False
    parsed = [statement for statement in statements if statement is not None]
    if not parsed:
        return False
    return all(isinstance(statement, STATEMENT_TYPES) for statement in parsed)
