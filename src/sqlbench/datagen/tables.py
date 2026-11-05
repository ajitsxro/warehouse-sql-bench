"""In-memory table data shared by the generator and the edge injectors."""

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from sqlbench.schema.spec import TableSpec


class Profile(StrEnum):
    MAIN = "main"
    EDGE = "edge"


@dataclass
class TableData:
    spec: TableSpec
    columns: dict[str, list[Any]]
    injectors: list[str] = field(default_factory=list)

    @property
    def row_count(self) -> int:
        first_column = next(iter(self.columns.values()))
        return len(first_column)
