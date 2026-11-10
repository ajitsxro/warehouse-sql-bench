from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from sqlbench.datagen.generator import find_integrity_problems, generate_tables
from sqlbench.datagen.tables import Profile
from sqlbench.datagen.values import generate_self_references
from sqlbench.schema.spec import SchemaSpec, load_schema

FIXTURE = Path(__file__).parent / "fixtures" / "mini_org.yaml"
SCHEMA = load_schema(FIXTURE)
MANAGER = SCHEMA.table("employees").column("manager_id")


def depth_of(key: Any, parent_of: dict[Any, Any]) -> int:
    depth = 0
    while parent_of[key] is not None:
        key = parent_of[key]
        depth += 1
    return depth


def test_self_reference_is_not_a_dependency_cycle() -> None:
    ordered = [table.name for table in SCHEMA.tables_in_dependency_order()]

    assert ordered.index("departments") < ordered.index("employees")


def test_self_reference_must_be_nullable() -> None:
    raw = SCHEMA.model_dump()
    raw["tables"][1]["columns"][1]["nullable"] = False

    with pytest.raises(ValidationError, match="self-references need nullable"):
        SchemaSpec.model_validate(raw)


def test_parents_are_earlier_rows_and_only_the_first_row_is_a_root() -> None:
    keys = list(range(1, 3001))
    parents = generate_self_references(MANAGER, keys, seed=1)

    assert parents[0] is None
    for key, parent in zip(keys[1:], parents[1:], strict=True):
        assert parent is not None and parent < key


def test_hierarchy_is_a_shallow_tree_with_few_managers() -> None:
    keys = list(range(1, 3001))
    parents = generate_self_references(MANAGER, keys, seed=1)
    parent_of = dict(zip(keys, parents, strict=True))

    depths = [depth_of(key, parent_of) for key in keys]
    managers = {parent for parent in parents if parent is not None}
    assert 3 <= max(depths) <= 10
    # Only the first sixth of rows can have reports.
    assert len(managers) <= len(keys) // 6


@pytest.mark.parametrize("profile", list(Profile))
def test_generated_hierarchy_passes_the_integrity_check(profile: Profile) -> None:
    tables = generate_tables(SCHEMA, profile, seed=3)

    assert find_integrity_problems(SCHEMA, tables) == []


def test_integrity_check_rejects_a_parent_on_a_later_row() -> None:
    tables = generate_tables(SCHEMA, Profile.MAIN, seed=3)
    tables["employees"].columns["manager_id"][5] = 3000

    problems = find_integrity_problems(SCHEMA, tables)

    assert problems == ["employees.manager_id: points at a later row"]


def test_edge_adds_roots_without_breaking_the_tree() -> None:
    tables = generate_tables(SCHEMA, Profile.EDGE, seed=3)
    employees = tables["employees"]
    parent_of = dict(
        zip(employees.columns["employee_id"], employees.columns["manager_id"], strict=True)
    )

    roots = [key for key, parent in parent_of.items() if parent is None]
    assert len(roots) > 1
    assert find_integrity_problems(SCHEMA, tables) == []
