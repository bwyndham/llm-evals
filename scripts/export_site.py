"""Export precomputed results for the portfolio page: one JSON file, no model calls.

Sources:
- Local runs (prompt comparisons, seeded regressions, A/A tests) are re-scored from
  their stored outputs with `eval rescore`, against a scratch copy of each store, so the
  numbers use today's checks and gate rules and the real store is left alone.
- CI runs come from their report artifacts, downloaded beforehand with
  `gh run download <run id> -n <artifact> -D .evals/site/ci/<name>`.
- Judge calibrations come from the spec-writing store's calibrations table.

Everything here is synthetic or public data, so the output is safe to publish.

Usage:  uv run python scripts/export_site.py --out <dir>

Writes <dir>/results.json (every number, small enough to ship with the page) and
<dir>/flips/<run id>.json (the before/after outputs of each run's flips, fetched on demand).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llm_evals import engine  # noqa: E402
from llm_evals.suite import load_suite  # noqa: E402

SUITES = {"spec-writing": ROOT / "suites" / "spec-writing", "research-agent": ROOT / "suites" / "research-agent"}
CI = ROOT / ".evals" / "site" / "ci"

# CI runs, in the order they happened.
CI_RUNS = [
    {"dir": "pr1-aa", "source": "CI · PR #1", "date": "2026-10-21", "role": "aa",
     "title": "First CI run: the same prompt on both arms",
     "note": "Both arms ran v2. The gate said HOLD: the difference was −1.1%, but at 3 samples per case "
             "it could only detect a 6.8-point drop."},
    {"dir": "pr3-spec-v3", "source": "CI · PR #3", "date": "2026-10-21", "role": "prompt",
     "title": "Spec writing v3: strict Gherkin and shorter titles",
     "note": "v3 targets v2's most common gating failures. First run at 6 samples per case."},
    {"dir": "pr7-research-v3", "source": "CI · PR #7", "date": "2026-10-21", "role": "prompt",
     "title": "Research agent v3: never pick a side in a conflict",
     "note": "v2 described most conflicts and then answered with one side anyway."},
]

# Local prompt comparisons worth showing.
LOCAL_COMPARES = [
    {"suite": "spec-writing", "run": "813fc8736e7941aeb7582c491a90a5a5", "date": "2026-10-19",
     "title": "Spec writing v2: one Feature, split bundles, flag gaps",
     "note": "The first prompt promoted through the gate."},
]

MAX_FLIPS = 12


def _rescore(suite: str, run_id: str, scratch: Path) -> dict[str, Any]:
    out = scratch / f"{suite}-{run_id[:8]}"
    db = scratch / f"{suite}.sqlite"
    proc = subprocess.run(
        ["uv", "run", "eval", "rescore", str(SUITES[suite]), "--run", run_id, "--db", str(db),
         "--out", str(out)],
        cwd=ROOT, capture_output=True, text=True)
    report = out / "report.json"
    if not report.exists():
        raise RuntimeError(f"rescore of {suite}/{run_id} wrote no report:\n{proc.stderr[-2000:]}")
    return json.loads(report.read_text())


def _case_text(suite: str) -> dict[str, str]:
    cases = engine.build_cases(load_suite(SUITES[suite]))
    return {c.id: str(c.input.get("ask") or c.input.get("question") or "") for c in cases}


def _version_name(suite: str, version_id: str) -> str:
    manifest = engine.find_manifest(load_suite(SUITES[suite]), version_id)
    return manifest.name if manifest else version_id


DISPLAY = {"baseline": "v1"}


def _slim(report: dict[str, Any], meta: dict[str, Any], asks: dict[str, dict[str, str]]) -> dict[str, Any]:
    suite = report["suite"]
    c = report["comparison"]
    b, d = report["baseline"], report["candidate"]
    gating = report.get("gating") or {}
    checks = [{"name": name, "gates": gating.get(name, "always") == "always",
               "baseline": b["check_pass_rates"].get(name), "candidate": d["check_pass_rates"].get(name)}
              for name in b.get("check_pass_rates", {})]
    flips = []
    for f in (report.get("flips") or [])[:MAX_FLIPS]:
        base_id = f["case_id"].split("~")[0]
        flips.append({k: f.get(k) for k in ("case_id", "direction", "baseline_rate", "candidate_rate",
                                            "strata", "failed_checks", "baseline_output", "candidate_output")}
                     | {"input": asks[suite].get(base_id, "")})
    return {
        "id": report["run_id"], "suite": suite, **meta,
        "decision": report["decision"], "tolerance": report["tolerance"],
        "n_cases": report["n_cases"], "samples_per_case": report["samples_per_case"],
        "delta": c["delta"], "ci_low": c["ci_low"], "ci_high": c["ci_high"], "mde": c["mde"],
        "mcnemar_p": c.get("mcnemar_p"),
        "baseline": {"name": DISPLAY.get(b["version_name"], b["version_name"]), "id": b["version_id"], "pass_rate": b["pass_rate"],
                     "cost_usd": b["cost_usd"], "p95_latency_ms": b.get("p95_latency_ms")},
        "candidate": {"name": DISPLAY.get(d["version_name"], d["version_name"]), "id": d["version_id"], "pass_rate": d["pass_rate"],
                      "cost_usd": d["cost_usd"], "p95_latency_ms": d.get("p95_latency_ms")},
        "checks": checks,
        "check_regressions": [{"check": r["check"], "delta": r["delta"], "ci_low": r["ci_low"],
                               "ci_high": r["ci_high"]} for r in report.get("check_regressions") or []],
        "critical_confirmed": [f"{x['case_id']}: {x['check']}" for x in report.get("critical_confirmations") or []
                               if x.get("confirmed")],
        "flips_total": len(report.get("flips") or []),
        "flips": flips,
    }


def _runs_of(suite: str, kinds: tuple[str, ...]) -> list[sqlite3.Row]:
    db = sqlite3.connect(SUITES[suite] / ".evals" / "evals.sqlite")
    db.row_factory = sqlite3.Row
    marks = ",".join("?" * len(kinds))
    return db.execute(f"SELECT id, kind, started_at, baseline_version_id, candidate_version_id FROM runs "
                      f"WHERE kind IN ({marks}) ORDER BY started_at", kinds).fetchall()


PROMPTS = {  # faithfulness judge prompt hash -> what changed
    "54a49cf3c38f": "Original prompt",
    "d071d868cb4f": "Elaborations allowed",
    "7eb1b8bbc34c": "Elaborations list + strict actor rule",
}


def _calibrations() -> list[dict[str, Any]]:
    """Each distinct (prompt, label set) calibration once, in the order it was computed."""
    db = sqlite3.connect(SUITES["spec-writing"] / ".evals" / "evals.sqlite")
    db.row_factory = sqlite3.Row
    out, seen = [], set()
    for row in db.execute("SELECT * FROM calibrations ORDER BY computed_at"):
        key = (row["prompt_hash"], row["n"], round(row["kappa"] or 0, 4), round(row["agreement"] or 0, 4))
        if key in seen:
            continue
        seen.add(key)
        out.append({k: row[k] for k in ("unit_key", "judge_model", "prompt_hash", "n", "agreement", "kappa",
                                        "sensitivity", "specificity", "approved", "computed_at")}
                   | {"prompt": PROMPTS.get(row["prompt_hash"], row["prompt_hash"])})
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True, help="Directory to write")
    args = parser.parse_args()

    asks = {s: _case_text(s) for s in SUITES}
    scratch = Path(tempfile.mkdtemp(prefix="llm-evals-site-"))
    for suite, path in SUITES.items():
        shutil.copy(path / ".evals" / "evals.sqlite", scratch / f"{suite}.sqlite")

    runs = []
    for spec in CI_RUNS:
        report = json.loads((CI / spec["dir"] / "report.json").read_text())
        runs.append(_slim(report, {k: v for k, v in spec.items() if k != "dir"}, asks))
    for spec in LOCAL_COMPARES:
        report = _rescore(spec["suite"], spec["run"], scratch)
        runs.append(_slim(report, {"source": "local", "role": "prompt", "date": spec["date"],
                                   "title": spec["title"], "note": spec["note"]}, asks))

    seeded: dict[tuple[str, str], dict[str, Any]] = {}
    for suite in SUITES:
        for row in _runs_of(suite, ("seeded", "seeded-kf")):
            name = _version_name(suite, row["candidate_version_id"])
            if suite == "spec-writing" and not name.endswith("_v2"):
                continue  # spec writing's v1 variants were superseded by v2's on the same 30 cases
            report = _rescore(suite, row["id"], scratch)
            if report["decision"] == "INCOMPLETE":
                continue  # a stored run with a different sample count; an earlier run of the variant stands
            variant = name.removeprefix("seeded_").removesuffix("_v2")
            note = "known-facts cases" if row["kind"] == "seeded-kf" else ""
            seeded[(suite, variant + note)] = _slim(report, {
                "source": "local", "role": "seeded", "date": row["started_at"][:10],
                "title": variant.replace("_", " "), "note": note}, asks)  # later runs replace earlier ones
    seeded_list = list(seeded.values())

    aa = {}
    for suite in SUITES:
        rows = _runs_of(suite, ("aa",))
        results = []
        for row in rows:
            r = _rescore(suite, row["id"], scratch)
            results.append({"delta": r["comparison"]["delta"], "mde": r["comparison"]["mde"],
                            "decision": r["decision"], "n_cases": r["n_cases"], "samples_per_case": r["samples_per_case"],
                            "critical_confirmed": len([x for x in r.get("critical_confirmations") or [] if x.get("confirmed")])})
        aa[suite] = results

    data = {
        "generated": "2026-10-21",
        "note": "Precomputed from synthetic and public eval data. No live model calls.",
        "runs": runs,
        "seeded": seeded_list,
        "aa": aa,
        "calibrations": _calibrations(),
    }
    flips_dir = args.out / "flips"
    flips_dir.mkdir(parents=True, exist_ok=True)
    for run in runs + seeded_list:
        outputs = {f["case_id"]: {"input": f.pop("input"), "baseline_output": f.pop("baseline_output"),
                                  "candidate_output": f.pop("candidate_output")} for f in run["flips"]}
        (flips_dir / f"{run['id']}.json").write_text(json.dumps(outputs, ensure_ascii=False))
    (args.out / "results.json").write_text(json.dumps(data, indent=1, ensure_ascii=False))
    print(f"wrote {args.out}: {len(runs)} runs, {len(seeded_list)} seeded, "
          f"{sum(len(v) for v in aa.values())} A/A, {len(data['calibrations'])} calibrations")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
