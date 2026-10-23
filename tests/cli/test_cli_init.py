"""`llm-evals init` writes a starter suite that runs offline straight away."""

from __future__ import annotations

import json
import subprocess

from typer.testing import CliRunner

from llm_evals.cli.main import app

runner = CliRunner()


def test_init_then_gate_offline(tmp_path):
    suite = tmp_path / "triage"
    res = runner.invoke(app, ["init", str(suite)])
    assert res.exit_code == 0, res.output
    for f in ("suite.yaml", "versions/v1.yaml", "versions/v2.yaml", "data/cases.jsonl",
              "schemas/output.schema.json", "plugins/checks.py", "plugins/mock.py", ".gitignore"):
        assert (suite / f).is_file(), f
    assert "suite: triage" in (suite / "suite.yaml").read_text()

    out = tmp_path / "report"
    res = runner.invoke(app, ["gate", str(suite), "--candidate", "versions/v2.yaml", "--provider", "mock",
                              "--skip-judges", "--out", str(out)])
    assert res.exit_code in (0, 1, 2), res.output  # a decision, not a crash or INCOMPLETE
    report = json.loads((out / "report.json").read_text())
    checks = report["candidate"]["check_pass_rates"]
    assert set(checks) == {"field_matches", "keywords_present", "max_length", "no_blame"}
    assert checks["no_blame"] == 1.0 and report["baseline"]["check_pass_rates"]["no_blame"] < 1.0


def test_init_refuses_a_folder_with_files_and_writes_ci_at_the_repo_root(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "evals").mkdir()
    (tmp_path / "evals" / "keep.txt").write_text("mine")
    res = runner.invoke(app, ["init", str(tmp_path / "evals")])
    assert res.exit_code != 0 and "isn't empty" in res.output

    res = runner.invoke(app, ["init", str(tmp_path / "evals" / "triage"), "--ci"])
    assert res.exit_code == 0, res.output
    workflow = (tmp_path / ".github" / "workflows" / "llm-evals.yml").read_text()
    assert "suite: evals/triage" in workflow and '"evals/triage/**"' in workflow
    assert "uses: bwyndham/llm-evals@v0.1.0" in workflow


def test_gate_against_a_base_checkout_with_the_same_plugins(tmp_path):
    """CI's case: the PR's suite and the base branch's copy both define the same
    plug-in checks. Reading the base copy's baseline must not load its plug-ins."""
    import shutil

    pr = tmp_path / "pr" / "triage"
    assert runner.invoke(app, ["init", str(pr)]).exit_code == 0
    base = tmp_path / "base" / "triage"
    shutil.copytree(pr, base)
    (pr / "versions" / "v1.yaml").write_text((pr / "versions" / "v2.yaml").read_text())  # the PR's prompt change
    out = tmp_path / "report"
    res = runner.invoke(app, ["gate", str(pr), "--baseline-dir", str(base), "--provider", "mock",
                              "--skip-judges", "--skip-unchanged", "--out", str(out)])
    assert res.exit_code in (0, 1, 2), res.output
    assert (out / "report.json").exists()


def test_init_ci_can_point_the_workflow_at_a_local_copy_of_the_action(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    res = runner.invoke(app, ["init", str(tmp_path / "evals" / "x"), "--ci",
                              "--action", "./.github/actions/llm-evals"])
    assert res.exit_code == 0, res.output
    workflow = (tmp_path / ".github" / "workflows" / "llm-evals.yml").read_text()
    assert "uses: ./.github/actions/llm-evals" in workflow and "__ACTION__" not in workflow
