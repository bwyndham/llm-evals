from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from agents_helpers import VARIANTS, system_prompt  # type: ignore[import-not-found]

from llm_evals.types import VersionManifest


@pytest.fixture
def prompts_dir(tmp_path: Path) -> Path:
    d = tmp_path / "prompts"
    d.mkdir()
    (d / "user.jinja").write_text("Question: {{ question }}\n")
    for name, kwargs in VARIANTS.items():
        (d / f"{name}.jinja").write_text(system_prompt(**kwargs))
    return d


@pytest.fixture
def make_version(prompts_dir: Path):
    def _make(variant: str = "baseline", **params: Any) -> VersionManifest:
        return VersionManifest(
            name=variant,
            model="claude-haiku-5-5",
            prompt_templates={
                "system": str(prompts_dir / f"{variant}.jinja"),
                "user": str(prompts_dir / "user.jinja"),
            },
            params=params,
            id=f"v-{variant}",
        )

    return _make


@pytest.fixture
def version(make_version) -> VersionManifest:
    return make_version()


@pytest.fixture
def memory_path(tmp_path: Path) -> Path:
    p = tmp_path / "world_facts.jsonl"
    facts = [
        {"subject": "Halvorsen Bridge", "attribute": "completion year", "value": "1931"},
        {"subject": "Vell River", "attribute": "length", "value": "240 kilometres"},
    ]
    p.write_text("".join(json.dumps(f) + "\n" for f in facts))
    return p
