"""Back up human labels to a file that can live in git, and restore them.

Labels are the expensive part of calibration (CAL-1): hours of human time,
while everything else can be recomputed. They live in the gitignored SQLite
store, keyed by run ids that only that store knows. A backup record therefore
carries the labeled output itself, so a restore into an empty store rebuilds
the samples and their labels; re-judging them (`rescore` or a re-judge script)
then gives fresh verdicts to calibrate against.

One record per (labeler, unit, version, case, output): the same output copied
into several runs (re-judges reuse stored outputs) is one label, and the most
recent verdict wins.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from llm_evals.store.sqlite import Store
from llm_evals.types import Result


def _output_hash(output: Any) -> str:
    return hashlib.sha256(json.dumps(output, sort_keys=True, default=str).encode()).hexdigest()[:16]


def export_labels(store: Store, *, unit_key: str | None = None) -> list[dict[str, Any]]:
    """Every distinct human label with the output it judged, oldest first."""
    latest: dict[tuple, dict[str, Any]] = {}
    for lab in store.labels_for(unit_key=unit_key):
        result = store.get_result(lab.run_id, lab.version_id, lab.case_id, lab.sample_idx)
        if result is None or result.output is None:
            continue
        digest = _output_hash(result.output)
        key = (lab.labeler, lab.unit_key, lab.version_id, lab.case_id, digest)
        record = {"labeler": lab.labeler, "unit_key": lab.unit_key, "label": lab.label,
                  "note": lab.note, "labeled_at": lab.labeled_at, "version_id": lab.version_id,
                  "case_id": lab.case_id, "output_sha": digest, "output": result.output}
        if key not in latest or record["labeled_at"] >= latest[key]["labeled_at"]:
            latest[key] = record
    return sorted(latest.values(), key=lambda r: (r["labeled_at"], r["case_id"], r["unit_key"]))


def write_labels(records: list[dict[str, Any]], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n" for r in records))
    return path


def read_labels(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def restore_labels(store: Store, records: list[dict[str, Any]], *, suite: str) -> tuple[str, int]:
    """Rebuild the labeled samples in a new run of kind "labelset" and save their
    labels there. Returns the run id and the number of labels saved."""
    run_id = store.create_run(suite=suite, kind="labelset", meta={"restored_labels": len(records)})
    run_id = run_id if isinstance(run_id, str) else run_id["id"]
    slots: dict[tuple[str, str, str], int] = {}  # (version, case, output) -> sample_idx in this run
    used: dict[tuple[str, str], int] = {}
    for rec in records:
        slot = (rec["version_id"], rec["case_id"], rec["output_sha"])
        if slot not in slots:
            idx = used.get((rec["version_id"], rec["case_id"]), 0)
            used[(rec["version_id"], rec["case_id"])] = idx + 1
            slots[slot] = idx
            store.save_result(Result(run_id=run_id, case_id=rec["case_id"], version_id=rec["version_id"],
                                     sample_idx=idx, output=rec["output"]))
        store.save_label(run_id, rec["version_id"], rec["case_id"], slots[slot], rec["unit_key"],
                         rec["label"], rec["labeler"], note=rec.get("note", ""),
                         labeled_at=rec.get("labeled_at") or None)
    return run_id, len(records)
