from __future__ import annotations

from pathlib import Path

import pytest

from llm_evals.runners.manifest import load_manifest
from llm_evals.types import Case, VersionManifest




@pytest.fixture
def suite_dir(tmp_path: Path) -> Path:
    d = tmp_path / "suite"
    (d / "prompts").mkdir(parents=True)
    (d / "prompts" / "system.j2").write_text("You write specs.\n")
    (d / "prompts" / "user.j2").write_text("Ask: {{ ask }}\n")
    (d / "versions").mkdir()
    (d / "versions" / "v1.yaml").write_text(
        "name: v1\n"
        "model: claude-haiku-5-5\n"
        "prompt_templates:\n"
        "  system: prompts/system.j2\n"
        "  user: prompts/user.j2\n"
        "params:\n"
        "  max_tokens: 4000\n"
        "  effort: low\n"
    )
    return d


@pytest.fixture
def version(suite_dir: Path) -> VersionManifest:
    return load_manifest(suite_dir / "versions" / "v1.yaml", suite_dir)


@pytest.fixture
def case() -> Case:
    return Case(id="c1", input={"ask": "let admins export usage"}, provenance="constructed")
