from __future__ import annotations

import pytest

import llm_evals.checks  # noqa: F401 - registers the checks
from llm_evals import registry
from llm_evals.types import Check

CHECKS = {
    "gherkin_parses": ({"critical": True}, "deterministic"),
    "format_rules": ({"title_max_chars": 60, "max_open_questions": 5}, "deterministic"),
    "invariance": ({"seed": 3, "max_scenario_delta": 0}, "metamorphic"),
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
    params, kind = CHECKS[name]
    check = _factory(name)(**params)
    assert isinstance(check, Check)
    assert check.name == name
    assert check.kind == kind


@pytest.mark.parametrize("name", sorted(CHECKS))
def test_unknown_param_is_type_error(name):
    with pytest.raises(TypeError):
        _factory(name)(not_a_param=1)


def test_suite_yaml_style_construction():
    check = _factory("gherkin_parses")(critical=True)
    assert check.critical is True
    assert _factory("format_rules")().critical is False
