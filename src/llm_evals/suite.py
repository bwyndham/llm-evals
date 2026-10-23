"""Suite config: one YAML file per workload (PRD: Pilot suites)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator


class RunnerConfig(BaseModel):
    type: str
    options: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_raw(cls, raw: dict[str, Any]) -> "RunnerConfig":
        raw = dict(raw)
        return cls(type=raw.pop("type"), options=raw)


class CheckConfig(BaseModel):
    name: str
    params: dict[str, Any] = Field(default_factory=dict)


class GateConfig(BaseModel):
    tolerance: float = 0.02
    max_infra_share: float = 0.05  # RUN-11: above this share a run is INCOMPLETE
    # STAT-9: two-sided alpha shared by the per-check regression tests (Bonferroni-split).
    per_check_alpha: float = 0.025
    # Critical flips are confirmed by rerunning the flagged case this many times per arm;
    # 0 disables confirmation (every unanimous flip fires).
    critical_confirm_k: int = 8
    critical_alpha: float = 0.05


class SuiteConfig(BaseModel):
    suite: str
    runner: RunnerConfig
    baseline: str  # manifest path, relative to the suite dir
    cases: list[dict[str, Any]]
    checks: list[CheckConfig]
    samples: int = 3
    gate: GateConfig = Field(default_factory=GateConfig)
    seeded_variants: list[str] = Field(default_factory=list)
    pairwise: dict[str, Any] | None = None  # CHK-7 config; report-only in v1
    judge: dict[str, Any] = Field(default_factory=lambda: {"model": "claude-sonnet-5-5"})
    triggers: list[str] = Field(default_factory=list)  # INT-6 path globs
    # The suite's own checks, judges, runners and schemas: module names or .py files
    # relative to the suite folder, imported before anything is built.
    plugins: list[str] = Field(default_factory=list)
    # Stand-in for the model under `--provider mock`: `file.py:function` or
    # `module:function`, called as fn(system=, user=, rng=) -> output dict.
    mock: str | None = None
    dir: Path = Path(".")

    @field_validator("checks", mode="before")
    @classmethod
    def _checks(cls, value: Any) -> Any:
        # YAML shape: `- gherkin_parses: {critical: true}` or `- gherkin_parses`
        out = []
        for item in value or []:
            if isinstance(item, str):
                out.append({"name": item})
            elif isinstance(item, dict) and len(item) == 1 and "name" not in item:
                (name, params), = item.items()
                out.append({"name": name, "params": params or {}})
            else:
                out.append(item)
        return out

    @field_validator("runner", mode="before")
    @classmethod
    def _runner(cls, value: Any) -> Any:
        if isinstance(value, dict) and "options" not in value:
            return RunnerConfig.from_raw(value)
        return value

    def path(self, relative: str) -> Path:
        return (self.dir / relative).resolve()


def baseline_of(path: Path) -> str:
    """The `baseline:` manifest path a suite.yaml names, read without loading the
    suite. In CI this reads the base branch's copy, whose plug-ins must not be
    imported alongside the PR's (they'd register the same names twice)."""
    path = Path(path)
    if path.is_dir():
        path = path / "suite.yaml"
    raw = yaml.safe_load(path.read_text()) or {}
    baseline = raw.get("baseline")
    if not isinstance(baseline, str) or not baseline:
        raise ValueError(f"{path} has no baseline: entry")
    return baseline


def load_suite(path: Path) -> SuiteConfig:
    path = Path(path)
    if path.is_dir():
        path = path / "suite.yaml"
    raw = yaml.safe_load(path.read_text())
    raw["dir"] = path.parent.resolve()
    suite = SuiteConfig.model_validate(raw)
    if suite.plugins:
        from llm_evals.registry import load_plugins

        load_plugins(suite.plugins, suite.dir)
    return suite
