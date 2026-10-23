"""`eval gate --skip-unchanged`: a PR that leaves the version alone makes no calls."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from llm_evals.cli.main import app

SUITE = Path(__file__).resolve().parents[2] / "suites" / "spec-writing"
runner = CliRunner()


def test_same_version_is_skipped_with_a_comment(tmp_path):
    out = tmp_path / "report"
    res = runner.invoke(app, ["gate", str(SUITE), "--baseline", "versions/v2.yaml",
                              "--candidate", "versions/v2.yaml", "--provider", "mock",
                              "--skip-unchanged", "--db", str(tmp_path / "s.sqlite"), "--out", str(out)])
    assert res.exit_code == 0, res.output
    comment = (out / "comment.md").read_text()
    assert comment.startswith("<!-- llm-evals:report:spec-writing -->") and "no version change" in comment
    assert not (out / "report.json").exists()  # nothing ran


def test_changed_version_still_runs(tmp_path):
    out = tmp_path / "report"
    res = runner.invoke(app, ["gate", str(SUITE), "--baseline", "versions/baseline.yaml",
                              "--candidate", "versions/v2.yaml", "--provider", "mock", "--limit", "2",
                              "--samples", "1", "--skip-judges", "--skip-unchanged",
                              "--db", str(tmp_path / "s.sqlite"), "--out", str(out)])
    assert res.exit_code in (0, 1, 2, 3), res.output
    assert (out / "report.json").exists()
    assert "no version change" not in (out / "comment.md").read_text()
