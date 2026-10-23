from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from llm_evals.cli.calibrate import calibrate_app
from llm_evals.store.sqlite import Store

from calibration_helpers import HASH_A, HASH_B, JUDGE_MODEL, SECRET, make_case, seed_run

runner = CliRunner()


def seeded_db(tmp_path: Path, *, save_cases: bool = True) -> tuple[Path, str]:
    db = tmp_path / "evals.sqlite"
    with Store(db) as store:
        verdicts = {"c1": {"checklist:a": False, "checklist:b": True},
                    "c2": {"checklist:a": True, "checklist:b": True}}
        run = seed_run(store, verdicts, run_id="run1", save_cases=save_cases)
    return db, run


def test_label_is_blind_and_saves_labels(tmp_path: Path) -> None:
    db, run = seeded_db(tmp_path)
    # 2 samples x 2 units; skip the last unit
    res = runner.invoke(calibrate_app, ["label", "--db", str(db), "--run", run, "--check",
                                        "checklist", "--n", "2", "--labeler", "ben"],
                        input="p\nx\nf\nf\ns\n")
    assert res.exit_code == 0, res.output
    out = res.output
    assert "ask text for c1" in out and "ask text for c2" in out
    assert "atom one of c1" in out and "atom two of c2" in out
    assert "Ticket for c1" in out
    assert "checklist:a: Is checklist:a satisfied?" in out  # rubric question, not verdict
    assert SECRET not in out and "judge" not in out.lower().replace("judge_model", "")
    assert "Answer p, f, s or q." in out  # the bad answer was re-asked
    assert "Saved 3 labels (1 skipped) as ben." in out
    with Store(db) as store:
        labels = store.labels_for(labeler="ben")
    assert sorted(lab.label for lab in labels) == ["fail", "fail", "pass"]
    # Rerun: the finished sample is skipped and only the skipped unit is asked.
    res = runner.invoke(calibrate_app, ["label", "--db", str(db), "--run", run, "--check",
                                        "checklist", "--n", "2", "--labeler", "ben",
                                        "--notes"], input="p\nlooks fine\n")
    assert res.exit_code == 0, res.output
    assert "Sample 1 of 1" in res.output and "Saved 1 labels" in res.output
    with Store(db) as store:
        assert {lab.note for lab in store.labels_for(labeler="ben")} == {"", "looks fine"}


def test_label_quit_keeps_what_was_answered(tmp_path: Path) -> None:
    db, run = seeded_db(tmp_path)
    res = runner.invoke(calibrate_app, ["label", "--db", str(db), "--run", run, "--check",
                                        "checklist", "--labeler", "ben"], input="p\nq\n")
    assert res.exit_code == 0, res.output
    assert "Saved 1 labels" in res.output


def test_label_needs_cases(tmp_path: Path) -> None:
    db, run = seeded_db(tmp_path, save_cases=False)
    args = ["label", "--db", str(db), "--run", run, "--check", "checklist", "--labeler", "ben"]
    res = runner.invoke(calibrate_app, args)
    assert res.exit_code == 1 and "--suite or --cases" in res.output
    cases = tmp_path / "cases.jsonl"
    cases.write_text("\n".join(make_case(c).model_dump_json() for c in ("c1", "c2")))
    res = runner.invoke(calibrate_app, [*args, "--cases", str(cases)], input="p\n" * 4)
    assert res.exit_code == 0, res.output
    assert "atom one of c1" in res.output


def test_missing_db(tmp_path: Path) -> None:
    res = runner.invoke(calibrate_app, ["status", "--db", str(tmp_path / "nope.sqlite")])
    assert res.exit_code == 1


def test_label_then_calibrate_end_to_end(tmp_path: Path) -> None:
    db = tmp_path / "evals.sqlite"
    with Store(db) as store:
        # 60 cases; checklist:a judge verdict alternates, checklist:b always passes;
        # faithfulness uses Score.prompt_hash (no per-unit hash).
        verdicts = {f"c{i:03d}": {"checklist:a": i % 2 == 0, "checklist:b": True}
                    for i in range(60)}
        seed_run(store, verdicts, run_id="run1", prompt_hash=HASH_B)
        seed_run(store, {f"c{i:03d}": {"faithfulness": i % 3 != 0} for i in range(60)},
                 check="faithfulness", run_id="run1", per_unit_hash=False)
    # Label 4 samples through the CLI, agreeing with the judge on checklist:a.
    res = runner.invoke(calibrate_app, ["label", "--db", str(db), "--run", "run1", "--check",
                                        "checklist", "--n", "4", "--labeler", "ben",
                                        "--seed", "3"], input="s\n" * 8)
    assert res.exit_code == 0 and "Saved 0 labels (8 skipped)" in res.output
    with Store(db) as store:
        for i in range(60):
            cid = f"c{i:03d}"
            store.save_label("run1", "v1", cid, 0, "checklist:a",
                             "pass" if i % 2 == 0 else "fail", "ben")
            store.save_label("run1", "v1", cid, 0, "checklist:b", "pass", "ben")
            # disagree with the judge on 2 of 60 faithfulness samples
            judge_pass = i % 3 != 0
            store.save_label("run1", "v1", cid, 0, "faithfulness",
                             "pass" if judge_pass != (i in (1, 3)) else "fail", "ben")

    res = runner.invoke(calibrate_app, ["calibrate", "--db", str(db), "--check", "checklist"])
    assert res.exit_code == 0, res.output
    rows = {line.split()[0]: line.split() for line in res.output.splitlines()
            if line.startswith("checklist:")}
    assert rows["checklist:a"][1:] == ["60", "1.000", "1.000", "1.000", "1.000", "yes"]
    assert rows["checklist:b"][1:] == ["60", "1.000", "n/a", "n/a", "1.000", "no"]

    res = runner.invoke(calibrate_app, ["calibrate", "--db", str(db), "--check",
                                        "faithfulness"])
    assert res.exit_code == 0, res.output
    with Store(db) as store:
        assert store.approved_calibration("checklist:a", JUDGE_MODEL, HASH_B)
        assert not store.approved_calibration("checklist:a", JUDGE_MODEL, HASH_A)
        assert not store.approved_calibration("checklist:b", JUDGE_MODEL, HASH_B)
        faith = store.latest_calibration("faithfulness", JUDGE_MODEL, HASH_A)
    # judge fails 20 (i%3==0); human flips i=1 (judge pass -> human fail) and i=3
    # (judge fail -> human pass): TP=19, FN=1, FP=1, TN=39
    assert (faith.tp, faith.fn, faith.fp, faith.tn) == (19, 1, 1, 39)
    assert faith.approved

    res = runner.invoke(calibrate_app, ["status", "--db", str(db)])
    assert res.exit_code == 0, res.output
    assert "faithfulness" in res.output and HASH_A in res.output and HASH_B in res.output
    assert res.output.count("checklist:") == 2


def test_calibrate_without_units(tmp_path: Path) -> None:
    db, _ = seeded_db(tmp_path)
    res = runner.invoke(calibrate_app, ["calibrate", "--db", str(db), "--check", "nope"])
    assert res.exit_code == 1
    res = runner.invoke(calibrate_app, ["status", "--db", str(db)])
    assert "No calibrations yet." in res.output
