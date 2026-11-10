"""Pydantic model for questions/<schema>/<id>.yaml.

Only questions with status verified are scored. Drafts and rejected questions
stay in the repo so their history is visible, but nothing downstream reads them.
"""

import datetime
from enum import StrEnum
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from sqlbench.warehouses.base import WarehouseName

ID_PATTERN = r"^[a-z][a-z0-9_]*$"


class Category(StrEnum):
    # One SQL text would work on both warehouses.
    PORTABLE = "portable"
    # The correct SQL differs by warehouse.
    DIALECT_SENSITIVE = "dialect_sensitive"


class Divergence(StrEnum):
    NONE = "none"
    # Same capability, different spelling (FLATTEN vs explode).
    SYNTAX = "syntax"
    # SQL that looks portable returns different results by default.
    SEMANTICS = "semantics"


class Feature(StrEnum):
    """What the gold SQL uses. Slices results; never affects scoring."""

    JOIN = "join"
    SELF_JOIN = "self_join"
    AS_OF_JOIN = "as_of_join"
    AGGREGATE = "aggregate"
    HAVING = "having"
    DISTINCT = "distinct"
    WINDOW = "window"
    QUALIFY = "qualify"
    CTE = "cte"
    RECURSIVE_CTE = "recursive_cte"
    SUBQUERY = "subquery"
    SET_OPERATION = "set_operation"
    CONDITIONAL = "conditional"
    DATE_MATH = "date_math"
    TIMEZONE = "timezone"
    STRING_FN = "string_fn"
    DECIMAL_MATH = "decimal_math"
    NULL_HANDLING = "null_handling"
    SEMI_STRUCTURED_PATH = "semi_structured_path"
    FLATTEN_EXPLODE = "flatten_explode"
    ARRAY_FN = "array_fn"
    HIGHER_ORDER_FN = "higher_order_fn"
    STRUCT_MAP_ACCESS = "struct_map_access"
    PIVOT = "pivot"


class Difficulty(StrEnum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class Status(StrEnum):
    DRAFT = "draft"
    VERIFIED = "verified"
    REJECTED = "rejected"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class GoldSql(StrictModel):
    snowflake: str = Field(min_length=1)
    databricks: str = Field(min_length=1)

    def for_warehouse(self, warehouse: WarehouseName) -> str:
        if warehouse == WarehouseName.SNOWFLAKE:
            return self.snowflake
        if warehouse == WarehouseName.DATABRICKS:
            return self.databricks
        raise ValueError(f"no gold SQL for {warehouse}")


class Question(StrictModel):
    id: str = Field(pattern=ID_PATTERN)
    # "schema" would shadow a BaseModel attribute, so the field has another name.
    schema_name: str = Field(alias="schema")
    question: str = Field(min_length=1)
    category: Category
    divergence: Divergence
    divergence_note: str | None = None
    features: list[Feature] = Field(min_length=1)
    difficulty: Difficulty
    gold_sql: GoldSql
    order_matters: bool
    float_tolerance: float = Field(ge=0)
    notes: str | None = None
    status: Status = Status.DRAFT
    verified_by: str | None = None
    verified_at: datetime.date | None = None

    @model_validator(mode="after")
    def check_divergence(self) -> "Question":
        portable = self.category == Category.PORTABLE
        if portable and self.divergence != Divergence.NONE:
            raise ValueError("a portable question has divergence: none")
        if not portable and self.divergence == Divergence.NONE:
            raise ValueError("a dialect_sensitive question needs divergence syntax or semantics")
        if self.divergence != Divergence.NONE and not self.divergence_note:
            raise ValueError("divergence_note is required unless divergence is none")
        return self

    @model_validator(mode="after")
    def check_verification(self) -> "Question":
        if self.status == Status.VERIFIED and (not self.verified_by or not self.verified_at):
            raise ValueError("a verified question needs verified_by and verified_at")
        return self


def load_question(path: Path) -> Question:
    return Question.model_validate(yaml.safe_load(path.read_text()))


def question_paths(questions_dir: Path, schema_name: str | None = None) -> list[Path]:
    """Every questions/<schema>/<id>.yaml, or one schema's."""
    if schema_name is not None:
        return sorted((questions_dir / schema_name).glob("*.yaml"))
    return sorted(questions_dir.glob("*/*.yaml"))
