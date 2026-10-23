"""`eval calibrate ...`: blind human labeling and judge calibration (CAL-1, CAL-2, CAL-4).

Mounted by the core CLI as a sub-app (`app.add_typer(calibrate_app, name="calibrate")`).

- `label` shows sampled outputs WITHOUT the judge's verdict and records a human
  pass/fail per gating unit.
- `calibrate` computes agreement for every (unit, judge model, prompt hash) seen in
  a check's stored scores. A unit's model and hash come from its sub_score entry
  (`judge_model`, `prompt_hash`), falling back to `Score.judge_model` /
  `Score.prompt_hash`.
- `status` lists the latest calibration of every triple.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import typer

from llm_evals.calibration import calibrate as run_calibration
from llm_evals.calibration import choose_samples, unit_triples
from llm_evals.store.sqlite import CalibrationRecord, Store
from llm_evals.types import Case

calibrate_app = typer.Typer(help="Label judge outputs blind and calibrate judge units.",
                            no_args_is_help=True)

_ANSWERS = {"p": "pass", "pass": "pass", "f": "fail", "fail": "fail",
            "s": "skip", "skip": "skip", "q": "quit", "quit": "quit"}


def _open(db: Path) -> Store:
    if str(db) != ":memory:" and not Path(db).exists():
        typer.echo(f"No database at {db}.", err=True)
        raise typer.Exit(1)
    return Store(db)


def _load_cases(store: Store, run_id: str, suite: Optional[Path], cases: Optional[Path]
                ) -> dict[str, Case]:
    found = store.run_cases(run_id)
    if suite is not None:
        from llm_evals import engine
        from llm_evals.suite import load_suite

        found.update({c.id: c for c in engine.build_cases(load_suite(suite),
                                                          include_held_out=True)})
    if cases is not None:
        from llm_evals.cases.loader import load_cases

        found.update({c.id: c for c in load_cases(cases)})
    return found


def _indent(text: str, prefix: str = "  ") -> str:
    return "\n".join(prefix + line for line in str(text).splitlines() or [""])


# What the judge is asked, so the human labels the same question (otherwise kappa
# measures a difference in definitions, not judge quality).
CRITERIA = {
    "faithfulness": """
How to label FAITHFULNESS (the same definition the judge uses):
  PASS only if both hold:
    1. Covered: every requirement that is NOT marked [HINTED] is stated in the ticket or
       tested by a scenario. Paraphrase counts. The actor matters only when the ask
       names it: if the ask says "account admins", a ticket that says "any user" doesn't
       cover it. If the ask never says who, a generic "user" is fine; only a specific
       different role stated as fact ("admins", "staff") fails.
    2. Nothing invented: nothing that changes what gets built beyond the ask. That means
       no new feature or behaviour, no new number, limit, format or permission rule the ask
       doesn't state or imply, and nothing that alters a stated requirement (a different
       actor, value or condition). Assumptions and open questions never count.
       Sensible elaborations are fine: setup and preconditions, error messages for a stated
       rule, negative cases of a stated rule (who can't, what's rejected), the natural
       completion or inverse of a stated action, and details that make a requirement
       concrete without changing it. Test: would an engineer building only what the ask
       says still build this? If yes, it's an elaboration.
       A [HINTED] requirement written as a firm scenario counts as invented; written as an
       assumption or question, it is fine.
  Otherwise FAIL. Ignore style, duplicated scenarios, title length and Gherkin formatting:
  other checks cover those.
""",
    "claim_support": """
How to label CLAIM_SUPPORT (the same definition the judge uses):
  PASS only if every claim is supported: the cited document states the claim or directly
  entails it. A claim with no citation, or citing a document that doesn't say it, fails.
""",
}


def _source_facts(case: Case) -> list[str]:
    """Facts the labeler needs: spec-writing atoms, research-agent facts."""
    source = case.source or {}
    lines = []
    for key in ("atoms", "facts"):
        for item in source.get(key) or []:
            if isinstance(item, dict):
                text = item.get("text", json.dumps(item, sort_keys=True))
                hinted = item.get("mode") == "hinted" or item.get("id") in (source.get("hinted_atom_ids") or [])
                tag = "  [HINTED: only implied; not required. Should appear as an assumption or open question, not a firm requirement]" if hinted else ""
                lines.append(f"{item['id']}: {text}{tag}" if item.get("id") else str(text))
            else:
                lines.append(str(item))
    return lines


def _show(case: Case, output: Any, idx: int, total: int) -> None:
    typer.echo(f"\n=== Sample {idx} of {total} " + "=" * 40)
    for key in ("ask", "question"):
        if key in case.input:
            typer.echo(f"Input ({key}):")
            typer.echo(_indent(case.input[key]))
            break
    else:
        typer.echo("Input:")
        typer.echo(_indent(json.dumps(case.input, indent=2, sort_keys=True)))
    facts = _source_facts(case)
    if facts:
        typer.echo("Source facts:")
        for fact in facts:
            typer.echo(f"  - {fact}")
    typer.echo("Output:")
    typer.echo(_indent(json.dumps(output, indent=2, ensure_ascii=False)))


def _questions(store: Store, run_id: str, check: str) -> dict[str, str]:
    """unit key -> rubric question, when the check's sub_scores carry one (checklist).
    Only the question is read; verdicts, answers and reasons are never shown."""
    out: dict[str, str] = {}
    for row in store.score_rows(check=check, run_id=run_id):
        for sub in row.score.sub_scores:
            if isinstance(sub, dict) and sub.get("unit") and isinstance(sub.get("question"), str):
                out.setdefault(str(sub["unit"]), sub["question"])
    return out


def _ask(unit: str, question: str | None = None) -> str:
    if question:
        typer.echo(f"{unit}: {question}")
    while True:
        raw = typer.prompt(f"{unit}: pass / fail / skip (p/f/s, q to quit)")
        answer = _ANSWERS.get(raw.strip().lower())
        if answer:
            return answer
        typer.echo("  Answer p, f, s or q.")


@calibrate_app.command()
def label(
    db: Path = typer.Option(..., help="SQLite results store"),
    run: str = typer.Option(..., "--run", help="Run id to sample from"),
    check: str = typer.Option(..., help="Judge check whose units to label"),
    n: int = typer.Option(50, help="Samples to label"),
    labeler: str = typer.Option(..., help="Your name; labels are kept per labeler"),
    seed: int = typer.Option(0, help="Sampling seed"),
    skip_held_out: bool = typer.Option(False, help="Skip held-out cases (CASE-7)"),
    suite: Optional[Path] = typer.Option(None, help="Suite to read cases from if the run "
                                                    "didn't store them"),
    cases: Optional[Path] = typer.Option(None, help="Case JSONL to read cases from"),
    notes: bool = typer.Option(False, help="Ask for an optional note after each sample"),
) -> None:
    """Label sampled outputs blind: the judge's verdict is never shown."""
    store = _open(db)
    by_id = _load_cases(store, run, suite, cases)
    picks = choose_samples(store, run, check, labeler=labeler, n=n, seed=seed,
                           skip_held_out=skip_held_out, cases=by_id if skip_held_out else None)
    if not picks:
        typer.echo(f"Nothing to label for check {check!r} in run {run}.")
        return
    missing = sorted({p.case_id for p in picks if p.case_id not in by_id})
    if missing:
        typer.echo(f"Cases not found for {len(missing)} sampled case ids (e.g. {missing[0]}); "
                   "pass --suite or --cases.", err=True)
        raise typer.Exit(1)

    done = {(lab.run_id, lab.version_id, lab.case_id, lab.sample_idx, lab.unit_key)
            for lab in store.labels_for(run_id=run, labeler=labeler)}
    questions = _questions(store, run, check)
    if check in CRITERIA:
        typer.echo(CRITERIA[check])
    saved = skipped = 0
    for i, pick in enumerate(picks, start=1):
        result = store.get_result(*pick.key)
        assert result is not None  # choose_samples only returns samples with results
        _show(by_id[pick.case_id], result.output, i, len(picks))
        pending = [u for u in pick.units if (*pick.key, u) not in done]
        answers: dict[str, str] = {}
        for unit in pending:
            answer = _ask(unit, questions.get(unit))
            if answer == "quit":
                break
            answers[unit] = answer
        note = ""
        if notes and any(a != "skip" for a in answers.values()):
            note = typer.prompt("Note (optional)", default="", show_default=False)
        for unit, answer in answers.items():
            if answer == "skip":
                skipped += 1
                continue
            store.save_label(*pick.key, unit, answer, labeler, note=note)
            saved += 1
        if len(answers) < len(pending):  # quit
            break
    typer.echo(f"\nSaved {saved} labels ({skipped} skipped) as {labeler}.")


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _table(records: list[CalibrationRecord], *, wide: bool = False) -> None:
    head = ["unit", "n", "agreement", "kappa", "sensitivity", "specificity", "approved"]
    if wide:
        head += ["judge_model", "prompt_hash", "computed_at"]
    rows = []
    for r in records:
        row = [r.unit_key, str(r.n), _fmt(r.agreement), _fmt(r.kappa), _fmt(r.sensitivity),
               _fmt(r.specificity), "yes" if r.approved else "no"]
        if wide:
            row += [r.judge_model, r.prompt_hash, r.computed_at]
        rows.append(row)
    widths = [max(len(x) for x in col) for col in zip(head, *rows)]
    for row in [head, *rows]:
        typer.echo("  ".join(x.ljust(w) for x, w in zip(row, widths)).rstrip())


@calibrate_app.command("calibrate")
def calibrate_cmd(
    db: Path = typer.Option(..., help="SQLite results store"),
    check: str = typer.Option(..., help="Judge check whose units to calibrate"),
    min_n: int = typer.Option(50, help="Labels needed to approve"),
    min_kappa: float = typer.Option(0.6, help="Cohen's kappa needed to approve"),
) -> None:
    """Compute human-judge agreement per unit (positive class = fail) and store it."""
    store = _open(db)
    triples = unit_triples(store, check)
    if not triples:
        typer.echo(f"No gating units with a judge model and prompt hash in check {check!r}.",
                   err=True)
        raise typer.Exit(1)
    records = [run_calibration(store, t.key, t.judge_model, t.prompt_hash,
                               min_n=min_n, min_kappa=min_kappa) for t in triples]
    _table(records, wide=len({t.key for t in triples}) != len(triples))  # a unit seen twice
    typer.echo(f"Bars: n >= {min_n}, kappa >= {min_kappa}. Sensitivity = judge catches "
               "human-labeled failures; specificity = judge passes human-labeled passes.")


@calibrate_app.command()
def serve(
    db: Path = typer.Option(..., help="SQLite results store"),
    run: str = typer.Option(..., help="Run whose judged samples to label"),
    check: str = typer.Option(..., help="Judge check to label (e.g. faithfulness)"),
    labeler: str = typer.Option(..., help="Your name; labels are kept per labeler"),
    n: int = typer.Option(50, help="Samples to label"),
    seed: int = typer.Option(0),
    suite: Optional[Path] = typer.Option(None, help="Suite to read cases from if the run "
                                                    "didn't store them"),
    port: int = typer.Option(8765),
) -> None:
    """Open a local labeling app in the browser (same blind rules as `label`)."""
    from llm_evals.calibration.server import LabelSession, serve as make_server

    store = _open(db)
    session = LabelSession(store=store, run_id=run, check=check, labeler=labeler,
                           cases=_load_cases(store, run, suite, None), target=n, seed=seed,
                           criteria=CRITERIA.get(check, ""))
    server = make_server(session, port=port)
    typer.echo(f"Labeling {len(session.samples)} samples of {check} at http://127.0.0.1:{port} "
               "(Ctrl+C to stop; every answer is saved as you go).")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        typer.echo("Stopped.")


@calibrate_app.command()
def export(
    db: Path = typer.Option(..., help="SQLite results store"),
    run: str = typer.Option(..., help="Run whose judged samples to label"),
    check: str = typer.Option(..., help="Judge check to label (e.g. faithfulness)"),
    labeler: str = typer.Option(..., help="Your name; labels are kept per labeler"),
    out: Path = typer.Option(..., help="Page data to write (JSON, safe to publish)"),
    keys: Path = typer.Option(..., help="Key map to write (JSON, keep it local)"),
    n: int = typer.Option(50, help="Samples to label"),
    seed: int = typer.Option(0),
    suite: Optional[Path] = typer.Option(None, help="Suite to read cases from if the run "
                                                    "didn't store them"),
) -> None:
    """Write the samples to label as blind page data, for labeling away from this machine."""
    from llm_evals.calibration.server import LabelSession, export_blind

    store = _open(db)
    session = LabelSession(store=store, run_id=run, check=check, labeler=labeler,
                           cases=_load_cases(store, run, suite, None), target=n, seed=seed,
                           criteria=CRITERIA.get(check, ""))
    data, key_map = export_blind(session)
    out.write_text(json.dumps(data, default=str))
    keys.write_text(json.dumps({"run_id": run, "check": check, "keys": key_map}))
    typer.echo(f"Wrote {len(data['samples'])} samples to {out} and their keys to {keys}.")


@calibrate_app.command("import")
def import_cmd(
    db: Path = typer.Option(..., help="SQLite results store"),
    labels: Path = typer.Option(..., help="Labels from the page: {sample id: {labels, note}}"),
    keys: Path = typer.Option(..., help="Key map written by `export`"),
    labeler: str = typer.Option(..., help="Whose labels these are"),
) -> None:
    """Save labels made on an exported page into the store."""
    from llm_evals.calibration.server import import_labels

    key_map = json.loads(keys.read_text())["keys"]
    counts = import_labels(_open(db), key_map, json.loads(labels.read_text()), labeler)
    typer.echo(f"Saved {counts['saved']} labeled samples; {counts['skipped']} skipped, "
               f"{counts['unknown']} with unknown ids.")


@calibrate_app.command("copy-labels")
def copy_labels_cmd(
    db: Path = typer.Option(..., help="SQLite results store"),
    from_run: str = typer.Option(..., "--from-run", help="Run the labels were made on"),
    to_run: str = typer.Option(..., "--to-run", help="Re-scored run of the same outputs"),
    labeler: str = typer.Option(..., help="Whose labels to copy"),
) -> None:
    """Carry labels to a re-judged run, so a changed judge prompt needs no relabeling."""
    from llm_evals.calibration.server import copy_labels

    counts = copy_labels(_open(db), from_run, to_run, labeler)
    typer.echo(f"Copied {counts['copied']} labels; {counts['different_output']} skipped because "
               f"the output differs, {counts['missing']} missing from a run.")


@calibrate_app.command("backup-labels")
def backup_labels(
    db: Path = typer.Option(..., help="SQLite results store"),
    out: Path = typer.Option(..., help="JSONL file to write (safe to commit: synthetic data only)"),
    unit: Optional[str] = typer.Option(None, help="Only this gating unit (default: all)"),
) -> None:
    """Write every human label, with the output it judged, to a file for git."""
    from llm_evals.calibration.backup import export_labels, write_labels

    records = export_labels(_open(db), unit_key=unit)
    write_labels(records, out)
    counts: dict[str, int] = {}
    for r in records:
        counts[r["label"]] = counts.get(r["label"], 0) + 1
    typer.echo(f"Wrote {len(records)} labels to {out} ({', '.join(f'{v} {k}' for k, v in sorted(counts.items()))}).")


@calibrate_app.command("restore-labels")
def restore_labels_cmd(
    db: Path = typer.Option(..., help="SQLite results store to restore into"),
    file: Path = typer.Option(..., "--file", help="JSONL written by backup-labels"),
    suite: str = typer.Option(..., help="Suite name to record on the restored run"),
) -> None:
    """Rebuild labeled samples and their labels from a backup, in a new run."""
    from llm_evals.calibration.backup import read_labels, restore_labels

    run_id, n = restore_labels(_open(db), read_labels(file), suite=suite)
    typer.echo(f"Restored {n} labels into run {run_id}. Re-judge that run to calibrate against them.")


@calibrate_app.command()
def status(
    db: Path = typer.Option(..., help="SQLite results store"),
) -> None:
    """List the latest calibration of every (unit, judge model, prompt hash)."""
    records = _open(db).latest_calibrations()
    if not records:
        typer.echo("No calibrations yet.")
        return
    _table(records, wide=True)
