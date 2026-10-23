"""REP-1/REP-2/INT-2: the Markdown report and the PR comment."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from llm_evals.engine import ComparisonReport, FlipDetail


def comment_marker(suite: str) -> str:
    """The hidden first line of a suite's PR comment. CI finds and updates the comment
    by it, so each suite keeps its own comment on a PR."""
    return f"<!-- llm-evals:report:{suite} -->"


_ICON = {"PROMOTE": "✅", "HOLD": "⏸️", "ROLLBACK": "⛔", "INCOMPLETE": "⚠️"}


def _pct(x: float) -> str:
    return f"{x:+.1%}" if x else "0.0%"


def _summary_lines(r: ComparisonReport) -> list[str]:
    c = r.comparison
    return [
        f"## {_ICON[r.decision.value]} {r.suite}: {r.decision.value}",
        "",
        f"Δ case pass rate **{_pct(c.delta)}** (95% CI {_pct(c.ci_low)} to {_pct(c.ci_high)}), "
        f"tolerance {r.tolerance:.1%}, {r.n_cases} cases × {r.samples_per_case} samples.",
        "",
        "| | Baseline | Candidate |",
        "| --- | --- | --- |",
        f"| Version | `{r.baseline.version_name}` ({r.baseline.version_id}) | "
        f"`{r.candidate.version_name}` ({r.candidate.version_id}) |",
        f"| Case pass rate | {r.baseline.pass_rate:.1%} | {r.candidate.pass_rate:.1%} |",
        f"| Cost | ${r.baseline.cost_usd:.4f} | ${r.candidate.cost_usd:.4f} |",
        f"| p95 latency | {r.baseline.p95_latency_ms:.0f} ms | {r.candidate.p95_latency_ms:.0f} ms |",
        f"| Infra errors | {r.baseline.infra_share:.1%} | {r.candidate.infra_share:.1%} |",
        *([f"| Judge spend (both arms) | ${r.judge_cost_usd:.4f} | |"] if r.judge_cost_usd else []),
    ]


def _flip_label(f: FlipDetail) -> str:
    strata = ", ".join(f"{k}={v}" for k, v in sorted(f.strata.items()))
    return f"`{f.case_id}` {f.baseline_rate:.0%} → {f.candidate_rate:.0%}" + (
        f" ({strata})" if strata else ""
    )


def render_comment(r: ComparisonReport, *, report_url: str | None = None, top: int = 5) -> str:
    """The sticky PR comment (INT-2). Held-out cases appear as ids only (REP-5)."""
    lines = [comment_marker(r.suite), *_summary_lines(r), ""]
    if r.check_regressions:
        lines += ["**Checks that regressed on their own:** " + ", ".join(
            f"`{cr.check}` {_pct(cr.delta)}" for cr in r.check_regressions), ""]
    if r.critical_new_failures:
        lines += ["**Critical checks newly failing:** "
                  + ", ".join(f"`{x}`" for x in r.critical_new_failures[:top]), ""]
    broke = [f for f in r.flips if f.direction == "broke"]
    fixed = [f for f in r.flips if f.direction == "fixed"]
    lines.append(f"**Flips:** {len(broke)} broke, {len(fixed)} fixed.")
    for f in broke[:top]:
        checks = ", ".join(f.failed_checks) or "n/a"
        lines.append(f"- broke {_flip_label(f)} — failing: {checks}")
    for f in fixed[:top]:
        lines.append(f"- fixed {_flip_label(f)}")
    for w in r.warnings:
        lines += ["", f"> {w}"]
    if report_url:
        lines += ["", f"[Full report]({report_url})"]
    return "\n".join(lines) + "\n"


def _block(value: Any) -> str:
    return "```json\n" + json.dumps(value, indent=2, ensure_ascii=False)[:4000] + "\n```"


def render_markdown(r: ComparisonReport) -> str:
    c = r.comparison
    lines = [*_summary_lines(r), ""]
    lines += [
        f"Minimum detectable Δ at 80% power: "
        f"{'n/a' if c.mde is None else f'{c.mde:.1%}'}. McNemar p = {c.mcnemar_p:.3f}.",
        f"Run `{r.run_id}` · case set `{r.case_set_hash[:12]}`.",
        "",
        "### Pass rate by check",
        "",
        "| Check | Baseline | Candidate | Δ |",
        "| --- | --- | --- | --- |",
    ]
    for name in sorted(set(r.baseline.check_pass_rates) | set(r.candidate.check_pass_rates)):
        b = r.baseline.check_pass_rates.get(name, 0.0)
        a = r.candidate.check_pass_rates.get(name, 0.0)
        lines.append(f"| {name} | {b:.1%} | {a:.1%} | {_pct(a - b)} |")
    if r.gating:
        lines += ["", "### What gates", ""]
        for name, units in sorted(r.gating.items()):
            if units == "always":
                lines.append(f"- `{name}`: gates (deterministic)")
            elif isinstance(units, str):
                lines.append(f"- `{name}`: {units}")
            elif units:
                lines.append(f"- `{name}`: gates through calibrated units {', '.join(units)}")
            else:
                lines.append(f"- `{name}`: reported only, not calibrated yet")
    if r.pairwise is not None:
        pw = r.pairwise
        lines += ["", "### Pairwise judge (report-only)", "",
                  f"Candidate preferred on {pw.candidate_wins} of {pw.cases} cases "
                  f"({pw.candidate_win_share:.0%}), baseline on {pw.baseline_wins}, ties {pw.ties}."]
    if r.check_regressions:
        lines += ["", "### Checks that regressed on their own (STAT-9)", ""]
        for cr in r.check_regressions:
            lines.append(f"- `{cr.check}`: Δ {_pct(cr.delta)} (CI {_pct(cr.ci_low)} to "
                         f"{_pct(cr.ci_high)}, α {cr.alpha:.3f})")
    if r.critical_confirmations:
        lines += ["", "### Critical flips checked by rerunning", ""]
        for cc in r.critical_confirmations:
            verdict = "confirmed" if cc.confirmed else "not confirmed (noise)"
            lines.append(f"- `{cc.case_id}` `{cc.check}`: baseline {cc.baseline_passes}/{cc.baseline_n}, "
                         f"candidate {cc.candidate_passes}/{cc.candidate_n} pass, "
                         f"p = {cc.p_value:.4f}, {verdict}")
    if r.critical_new_failures:
        lines += ["", "### Critical checks newly failing", ""]
        lines += [f"- `{x}`" for x in r.critical_new_failures]
    lines += ["", f"### Flips ({len(r.flips)})", ""]
    for f in r.flips:
        lines.append(f"#### {f.direction}: {_flip_label(f)}")
        if f.failed_checks:
            lines.append(f"Failing checks: {', '.join(f.failed_checks)}")
        if f.held_out:
            lines.append("_Held-out case: outputs not shown._")
        else:
            lines += ["", "Baseline:", _block(f.baseline_output), "", "Candidate:",
                      _block(f.candidate_output)]
        lines.append("")
    if r.warnings:
        lines += ["### Warnings", ""] + [f"- {w}" for w in r.warnings]
    return "\n".join(lines) + "\n"


def render_unchanged(suite: str, version: Any) -> str:
    """The PR comment when the PR leaves the version under test unchanged: nothing ran."""
    return "\n".join([
        comment_marker(suite),
        f"## ⏭️ {suite}: no version change",
        "",
        f"This PR doesn't change the version under test (`{version.name}`, {version.id}): same "
        "model, params and prompt templates as the base branch. The paid run was skipped.",
        "",
        "To check a change to cases or checks, re-score a stored run locally with `eval rescore`.",
        "",
    ])


def write_reports(r: ComparisonReport, out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "json": out_dir / "report.json",
        "markdown": out_dir / "report.md",
        "comment": out_dir / "comment.md",
    }
    paths["json"].write_text(r.model_dump_json(indent=2))
    paths["markdown"].write_text(render_markdown(r))
    paths["comment"].write_text(render_comment(r))
    return paths
