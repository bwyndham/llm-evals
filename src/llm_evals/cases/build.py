"""Regenerate a suite's constructed cases from its suite.yaml.

    uv run python -m llm_evals.cases.build suites/spec-writing

Reads every `cases` entry with a `render` key, renders it, and writes all
rendered cases to `<suite_dir>/data/cases.jsonl` (canonical JSONL).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import yaml

from llm_evals.cases.loader import case_set_hash, write_cases
from llm_evals.cases.renderers.ask_renderer import AskRenderer
from llm_evals.types import Case

CASES_FILE = Path("data/cases.jsonl")


def _render_entry(suite_dir: Path, entry: dict[str, Any]) -> list[Case]:
    name = entry["render"]
    if name != "ask_renderer":
        raise ValueError(f"build_constructed_cases does not know renderer {name!r}")
    renderer = AskRenderer(suite_dir / entry["atoms_path"])
    return renderer.render_suite(int(entry["n"]), int(entry["seed"]), entry.get("features"))


def build_constructed_cases(suite_dir: Path) -> Path:
    """Render the suite's constructed cases into data/cases.jsonl and return that path."""
    suite_dir = Path(suite_dir)
    spec = yaml.safe_load((suite_dir / "suite.yaml").read_text(encoding="utf-8")) or {}
    cases: list[Case] = []
    for entry in spec.get("cases") or []:
        if isinstance(entry, dict) and "render" in entry:
            cases.extend(_render_entry(suite_dir, entry))
    if not cases:
        raise ValueError(f"{suite_dir / 'suite.yaml'} has no `render` case entries")
    return write_cases(suite_dir / CASES_FILE, cases)


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    suite_dir = Path(args[0]) if args else Path("suites/spec-writing")
    out = build_constructed_cases(suite_dir)
    from llm_evals.cases.loader import load_cases

    cases = load_cases(out)
    print(f"wrote {len(cases)} cases to {out} (case set hash {case_set_hash(cases)[:12]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
