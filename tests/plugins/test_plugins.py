"""A suite's own plug-ins load from suite.yaml, as a file or a module next to it."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from llm_evals import registry
from llm_evals.suite import load_suite

CHECK = '''
from llm_evals.api import BaseCheck, Score, register_check


class Shouts(BaseCheck):
    name = "{name}"
    kind = "deterministic"

    def _score(self, case, output, ctx):
        text = str(output.get("reply", ""))
        if text.isupper():
            return self.fail("the reply is all capitals")
        return Score(check=self.name, kind=self.kind, passed=True, reason="ok")


@register_check("{name}")
def make(*, critical: bool = False):
    return Shouts(critical=critical)
'''


def _suite(tmp_path: Path, plugins: list[str]) -> Path:
    (tmp_path / "suite.yaml").write_text(textwrap.dedent(f"""
        suite: demo
        runner: {{type: prompt}}
        baseline: versions/v1.yaml
        cases: []
        checks: []
        plugins: {plugins}
    """))
    return tmp_path


def test_file_plugin_registers_its_check(tmp_path):
    (tmp_path / "plugins").mkdir()
    (tmp_path / "plugins" / "checks.py").write_text(CHECK.format(name="no_shouting_file"))
    load_suite(_suite(tmp_path, ["plugins/checks.py"]))
    check = registry.get("check", "no_shouting_file")()
    assert check.name == "no_shouting_file"
    load_suite(tmp_path)  # loading the suite again doesn't re-register (which would raise)


def test_module_plugin_next_to_the_suite(tmp_path):
    (tmp_path / "my_suite_checks.py").write_text(CHECK.format(name="no_shouting_module"))
    load_suite(_suite(tmp_path, ["my_suite_checks"]))
    assert "no_shouting_module" in registry.names("check")


def test_missing_plugin_says_where_it_looked(tmp_path):
    with pytest.raises(FileNotFoundError, match="plug-in file not found"):
        load_suite(_suite(tmp_path, ["plugins/nope.py"]))
    with pytest.raises(ModuleNotFoundError, match="put it next to suite.yaml"):
        load_suite(_suite(tmp_path, ["not_a_real_module_xyz"]))


def test_two_suites_may_share_a_plugin_name_but_not_take_a_builtin(tmp_path):
    for n in ("one", "two"):
        d = tmp_path / n
        d.mkdir()
        (d / "checks.py").write_text(CHECK.format(name="shared_plugin_check"))
        load_suite(_suite(d, ["checks.py"]))  # the second suite's copy replaces the first's
    assert "shared_plugin_check" in registry.names("check")
    clash = tmp_path / "clash"
    clash.mkdir()
    (clash / "checks.py").write_text(CHECK.format(name="field_matches"))
    with pytest.raises(ValueError, match="already registered"):
        load_suite(_suite(clash, ["checks.py"]))
