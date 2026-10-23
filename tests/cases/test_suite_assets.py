import json
from pathlib import Path

import pytest
import yaml
from jinja2 import Environment, StrictUndefined

from llm_evals.cases import case_set_hash, load_cases
from llm_evals.cases.renderers.ask_renderer import load_atom_sets
from llm_evals.types import VersionManifest


def test_atoms_load(suite_dir: Path) -> None:
    sets = load_atom_sets(suite_dir / "data" / "atoms.jsonl")
    assert len(sets) == 40
    assert len({s.id for s in sets}) == 40
    assert len({s.domain for s in sets}) >= 8
    kinds = {a.kind for s in sets for a in s.atoms}
    assert kinds == {"functional", "constraint", "non_functional"}
    for s in sets:
        assert 2 <= len(s.atoms) <= 5
        assert any(a.hint for a in s.atoms), s.id
        for a in s.atoms:
            for kw in a.keywords:
                assert kw in a.say.lower() and kw in a.vague.lower(), (s.id, a.id, kw)


def test_atom_lines_match_contract_shape(suite_dir: Path) -> None:
    for line in (suite_dir / "data" / "atoms.jsonl").read_text().splitlines():
        row = json.loads(line)
        assert {"id", "domain", "actor", "atoms"} <= set(row)
        for atom in row["atoms"]:
            assert {"id", "text", "kind"} <= set(atom)


MANIFESTS = ["baseline.yaml", "seeded_no_split_bundles.yaml", "seeded_no_assumption_flags.yaml"]


@pytest.mark.parametrize("name", MANIFESTS)
def test_manifests_parse_and_reference_templates(suite_dir: Path, name: str) -> None:
    data = yaml.safe_load((suite_dir / "versions" / name).read_text())
    assert "id" not in data
    manifest = VersionManifest.model_validate(data)
    assert manifest.model == "claude-haiku-5-5"
    assert manifest.params == {"max_tokens": 4000, "effort": "low"}
    assert set(manifest.prompt_templates) == {"system", "user"}
    for rel in manifest.prompt_templates.values():
        assert (suite_dir / rel).is_file(), rel
    assert manifest.name == name.removesuffix(".yaml")


def _templates(suite_dir: Path) -> list[Path]:
    return sorted((suite_dir / "prompts").rglob("*.jinja"))


def test_templates_render_with_strict_undefined(suite_dir: Path) -> None:
    env = Environment(undefined=StrictUndefined)
    paths = _templates(suite_dir)
    assert len(paths) == 8  # v1, v2 and v3 system prompts, the user prompt, four seeded variants
    for path in paths:
        out = env.from_string(path.read_text()).render(ask="hey team, please add CSV export")
        assert out.strip()
    user = env.from_string((suite_dir / "prompts" / "spec_user.jinja").read_text()).render(ask="MARKER-123")
    assert "MARKER-123" in user


def test_seeded_prompts_differ_from_baseline(suite_dir: Path) -> None:
    base = (suite_dir / "prompts" / "spec_system.jinja").read_text()
    no_split = (suite_dir / "prompts" / "seeded" / "no_split_bundles.jinja").read_text()
    no_flags = (suite_dir / "prompts" / "seeded" / "no_assumption_flags.jinja").read_text()
    assert "bundles" in base and "bundles" not in no_split
    assert "single" in no_split
    assert "Write an assumption instead" in base and "Write an assumption instead" not in no_flags
    for text in (base, no_split, no_flags):
        assert "80 characters" in text and "open_questions" in text


def test_suite_yaml(suite_dir: Path) -> None:
    spec = yaml.safe_load((suite_dir / "suite.yaml").read_text())
    assert spec["suite"] == "spec-writing"
    assert (suite_dir / spec["baseline"]).is_file()
    for rel in spec["seeded_variants"]:
        assert (suite_dir / rel).is_file()
    entry = spec["cases"][0]
    assert entry["render"] == "ask_renderer" and (suite_dir / entry["atoms_path"]).is_file()


def test_checked_in_cases_are_current(suite_dir: Path, tmp_path: Path) -> None:
    """data/cases.jsonl matches what build_constructed_cases produces now."""
    import shutil

    from llm_evals.cases.build import build_constructed_cases

    checked_in = load_cases(suite_dir / "data" / "cases.jsonl")
    assert len(checked_in) == 120
    copy = tmp_path / "suite"
    shutil.copytree(suite_dir, copy, ignore=shutil.ignore_patterns("cases.jsonl"))
    rebuilt = build_constructed_cases(copy)
    assert rebuilt.read_bytes() == (suite_dir / "data" / "cases.jsonl").read_bytes()
    assert case_set_hash(load_cases(rebuilt)) == case_set_hash(checked_in)
