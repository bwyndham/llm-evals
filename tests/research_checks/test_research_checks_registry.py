from __future__ import annotations

import pytest

import llm_evals.checks  # noqa: F401 - registers the checks
from llm_evals import registry
from llm_evals.types import Check

CHECKS = {
    "citations_exist": ({}, "structural", True),
    "answer_match": ({}, "reference", False),
    "follows_corpus": ({}, "metamorphic", False),
    "uncited_claims": ({"max_share": 0.2}, "structural", False),
    "claim_support": ({"suite_dir": "suites/research-agent"}, "faithfulness", False),
}


def _factory(name: str):
    """Prefer the public lookup; fall back to the raw registry if a sibling package
    imported by load_builtins() is broken mid-development."""
    try:
        return registry.get("check", name)
    except Exception:  # noqa: BLE001
        return registry._REGISTRIES["check"][name]


@pytest.mark.parametrize("name", sorted(CHECKS))
def test_registered_and_satisfies_protocol(name):
    assert name in registry._REGISTRIES["check"]
    params, kind, critical = CHECKS[name]
    check = _factory(name)(**params)
    assert isinstance(check, Check)
    assert (check.name, check.kind, check.critical) == (name, kind, critical)


@pytest.mark.parametrize("name", sorted(CHECKS))
def test_unknown_param_is_type_error(name):
    with pytest.raises(TypeError):
        _factory(name)(not_a_param=1)


def test_only_claim_support_needs_a_judge():
    needs = {name: getattr(_factory(name)(), "needs_judge", False) for name in CHECKS}
    assert needs == {name: name == "claim_support" for name in CHECKS}
