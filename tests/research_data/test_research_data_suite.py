from pathlib import Path

import pytest
import yaml
from jinja2 import Environment, StrictUndefined

from llm_evals.runners.manifest import load_manifest
from llm_evals.suite import load_suite
from llm_evals.types import VersionManifest

SUITE_DIR = Path(__file__).resolve().parents[2] / "suites" / "research-agent"
MANIFESTS = ["baseline.yaml", "seeded_ignore_corpus.yaml", "seeded_no_abstain.yaml",
             "seeded_drop_citations.yaml"]


def _read(rel: str) -> str:
    return (SUITE_DIR / rel).read_text()


@pytest.mark.parametrize("name", MANIFESTS)
def test_manifests(name: str) -> None:
    data = yaml.safe_load(_read(f"versions/{name}"))
    assert "id" not in data
    m = VersionManifest.model_validate(data)
    assert m.name == name.removesuffix(".yaml")
    assert m.model == "claude-haiku-5-5"
    assert m.params == {"max_tokens": 4000, "effort": "low"}
    assert set(m.prompt_templates) == {"system", "user"}
    assert m.prompt_templates["user"] == "prompts/research_user.jinja"
    loaded = load_manifest(SUITE_DIR / "versions" / name, SUITE_DIR)
    assert loaded.id


def test_manifest_ids_differ() -> None:
    ids = {load_manifest(SUITE_DIR / "versions" / n, SUITE_DIR).id for n in MANIFESTS}
    assert len(ids) == len(MANIFESTS)


def test_templates_render_with_strict_undefined() -> None:
    env = Environment(undefined=StrictUndefined)
    paths = sorted((SUITE_DIR / "prompts").rglob("*.jinja"))
    assert len(paths) == 10  # v1, v2 and v3 system prompts, user prompt, 3 v1 and 3 v2 seeded variants
    for path in paths:
        assert env.from_string(path.read_text()).render(question="In what year?").strip()
    user = env.from_string(_read("prompts/research_user.jinja")).render(question="MARKER-123")
    assert "MARKER-123" in user


def test_seeded_prompts_differ_from_baseline() -> None:
    base = _read("prompts/research_system.jinja")
    ignore = _read("prompts/seeded/ignore_corpus.jinja")
    no_abstain = _read("prompts/seeded/no_abstain.jinja")
    no_cite = _read("prompts/seeded/drop_citations.jinja")
    assert "Answer only from the documents" in base
    assert "Answer only from the documents" not in ignore and "general knowledge" in ignore
    assert '"not_found"' in base and "not_found" not in no_abstain and "best answer" in no_abstain
    assert "Every claim must list the doc ids" in base
    assert "Every claim must list the doc ids" not in no_cite and "optional" in no_cite and "doc_id" not in no_cite.split("{\n")[0]
    for text in (base, ignore, no_abstain, no_cite):
        for key in ('"status"', '"answer"', '"claims"', '"citations"', '"conflicts"', "search(", "fetch("):
            assert key in text
    assert "ONLY this JSON object" in base


def test_suite_yaml() -> None:
    suite = load_suite(SUITE_DIR)
    assert suite.suite == "research-agent"
    assert suite.runner.type == "agent_replay"
    assert suite.runner.options == {"entry": "llm_evals.agents.research:run", "on_unrecorded_call": "fail",
                                    "max_turns": 8}
    assert [c.name for c in suite.checks] == ["citations_exist", "answer_correct", "abstains_when_missing", "flags_conflicts", "follows_corpus",
                                              "uncited_claims", "claim_support"]
    assert suite.checks[0].params == {"critical": True}
    assert next(c for c in suite.checks if c.name == "uncited_claims").params == {"max_share": 0.1}
    assert suite.samples == 3 and suite.gate.tolerance == 0.05  # set from the A/A test
    assert (SUITE_DIR / suite.baseline).is_file()
    assert suite.cases == [{"from": "data/cases.jsonl"}, {"from": "data/known_facts_cases.jsonl"}]
    assert (SUITE_DIR / "data" / "cases.jsonl").is_file()
    assert len(suite.seeded_variants) == 3
    for rel in suite.seeded_variants:
        assert (SUITE_DIR / rel).is_file()
