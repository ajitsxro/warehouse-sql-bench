"""Run a question's gold SQL on both warehouses and both datasets, and compare.

Each gold runs on its own warehouse, against main and against edge: four
runs. Main and edge hold different data, so results are compared across
warehouses within a dataset: Snowflake main against Databricks main, and
the same for edge.
"""

from dataclasses import dataclass, field

from sqlbench.datagen.tables import Profile
from sqlbench.questions.compare import first_difference, normalize_rows
from sqlbench.questions.model import Question
from sqlbench.warehouses.base import QueryResult, Warehouse, WarehouseName

GOLD_WAREHOUSES = (WarehouseName.SNOWFLAKE, WarehouseName.DATABRICKS)
PROFILES = (Profile.MAIN, Profile.EDGE)
GOLD_TIMEOUT_S = 60

# One read-only adapter per warehouse and dataset, each with its session schema set.
GoldTargets = dict[tuple[WarehouseName, Profile], Warehouse]


@dataclass(frozen=True)
class GoldRun:
    warehouse: WarehouseName
    profile: Profile
    result: QueryResult


@dataclass
class GoldCheck:
    runs: list[GoldRun] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.failures

    def run(self, warehouse: WarehouseName, profile: Profile) -> QueryResult:
        for gold_run in self.runs:
            if gold_run.warehouse == warehouse and gold_run.profile == profile:
                return gold_run.result
        raise KeyError((warehouse, profile))


def check_question(question: Question, targets: GoldTargets) -> GoldCheck:
    check = GoldCheck()
    for warehouse in GOLD_WAREHOUSES:
        sql = question.gold_sql.for_warehouse(warehouse)
        for profile in PROFILES:
            result = targets[(warehouse, profile)].execute(sql, GOLD_TIMEOUT_S)
            check.runs.append(GoldRun(warehouse, profile, result))
            if result.error is not None:
                check.failures.append(
                    f"{warehouse}/{profile}: {result.error.category}: {result.error.message}"
                )
    if check.failures:
        return check

    for profile in PROFILES:
        normalized = []
        for warehouse in GOLD_WAREHOUSES:
            rows = check.run(warehouse, profile).rows
            normalized.append(normalize_rows(rows, question.order_matters))
        difference = first_difference(normalized[0], normalized[1], question.float_tolerance)
        if difference is not None:
            check.failures.append(f"{profile}: snowflake and databricks differ: {difference}")

    for warehouse in GOLD_WAREHOUSES:
        main_rows = check.run(warehouse, Profile.MAIN).rows
        edge_rows = check.run(warehouse, Profile.EDGE).rows
        if main_rows and not edge_rows:
            check.failures.append(
                f"{warehouse}: edge result is empty but main is not; the question may be ill-posed"
            )
    return check
