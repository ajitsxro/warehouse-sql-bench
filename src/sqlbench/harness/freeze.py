"""The frozen benchmark inputs: which questions, prompt, data and scorer a v1.0 run must use.

Every hash here is the same one a run records in its fingerprint, so a
stored run can be checked against the manifest field by field.
"""

import json
from pathlib import Path

from sqlbench.harness.executor import data_sha256
from sqlbench.harness.run import data_manifest_sha256, question_set_sha256
from sqlbench.questions.check import PROFILES
from sqlbench.questions.model import Question


def build_manifest(
    verified: list[tuple[Path, Question]],
    prompt_file: Path,
    prompt_sha256: str,
    manifest_dir: Path,
    scorer_version: int,
) -> dict[str, object]:
    """The manifest for the given verified questions; raises ValueError if there are none."""
    if not verified:
        raise ValueError("no verified questions to freeze")
    schema_names = sorted({question.schema_name for _, question in verified})
    data: dict[str, dict[str, str]] = {}
    for schema_name in schema_names:
        data[schema_name] = {}
        for profile in PROFILES:
            data[schema_name][profile.value] = data_sha256(manifest_dir, schema_name, profile)
    return {
        "scorer_version": scorer_version,
        "prompt": {"file": prompt_file.as_posix(), "sha256": prompt_sha256},
        "question_set_sha256": question_set_sha256([path for path, _ in verified]),
        "question_ids": sorted(question.id for _, question in verified),
        "data_manifest_sha256": data_manifest_sha256(manifest_dir, schema_names),
        "data": data,
    }


def write_manifest(manifest: dict[str, object], path: Path) -> None:
    path.write_text(json.dumps(manifest, indent=2) + "\n")
