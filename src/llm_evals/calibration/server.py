"""A local labeling app for judge calibration (CAL-1), served by `eval calibration serve`.

Same rules as the terminal labeler: samples are chosen by `choose_samples`
(stratified by judge verdict, deterministic), the judge's verdict, reason and the
version that produced an output are never sent to the page, and every answer is
written to the store as soon as it's given. Stdlib only: a threaded HTTP server
on localhost with a small JSON API and one HTML page.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from llm_evals.calibration import calibrate as run_calibration
from llm_evals.calibration import choose_samples, unit_triples
from llm_evals.store.sqlite import Store
from llm_evals.types import Case

PAGE = Path(__file__).with_name("static") / "label.html"


@dataclass
class LabelSession:
    store: Store
    run_id: str
    check: str
    labeler: str
    cases: dict[str, Case]
    target: int = 50
    seed: int = 0
    criteria: str = ""
    samples: list[dict[str, Any]] = field(default_factory=list)
    skipped: set[tuple] = field(default_factory=set)  # skipped this session; back next time
    lock: threading.Lock = field(default_factory=threading.Lock)

    def load(self) -> None:
        """The samples still to label: rebuilt on every page load, so a refresh never
        brings back one you already answered or skipped."""
        remaining = max(0, self.target - self.done_count())
        picks = choose_samples(self.store, self.run_id, self.check, labeler=self.labeler,
                               n=remaining + len(self.skipped), seed=self.seed)
        picks = [p for p in picks if tuple(p.key) not in self.skipped][:remaining]
        questions = _questions(self.store, self.run_id, self.check)
        self.samples = []
        for pick in picks:
            case = self.cases.get(pick.case_id)
            result = self.store.get_result(*pick.key)
            if case is None or result is None:
                continue
            self.samples.append({
                "key": list(pick.key),
                "case": _case_view(case),
                "output": result.output,
                "units": [{"unit": u, "question": questions.get(u)} for u in pick.units],
            })

    def check_units(self) -> set[str]:
        if not hasattr(self, "_units"):
            self._units = {t.key for t in unit_triples(self.store, self.check)}
        return self._units

    def done_count(self) -> int:
        """Samples this labeler has labeled (any unit) for this run and check."""
        units = self.check_units()
        keys = set()
        for lab in self.store.labels_for(run_id=self.run_id, labeler=self.labeler):
            if lab.unit_key in units:
                keys.add((lab.version_id, lab.case_id, lab.sample_idx))
        return len(keys)

    def seen_cases(self) -> list[str]:
        """Cases this labeler already labeled some output of (repeats get a note)."""
        units = self.check_units()
        return sorted({lab.case_id for lab in self.store.labels_for(run_id=self.run_id,
                                                                    labeler=self.labeler)
                       if lab.unit_key in units})

    def state(self) -> dict[str, Any]:
        with self.lock:
            self.load()
        return {"check": self.check, "labeler": self.labeler, "run_id": self.run_id,
                "target": self.target, "already_done": self.done_count(),
                "seen_cases": self.seen_cases(), "skipped": len(self.skipped),
                "criteria": self.criteria, "samples": self.samples}

    def skip(self, body: dict[str, Any]) -> dict[str, Any]:
        with self.lock:
            self.skipped.add(tuple(body["key"]))
        return {"ok": True, "skipped": len(self.skipped)}

    def save(self, body: dict[str, Any]) -> dict[str, Any]:
        run_id, version_id, case_id, sample_idx = body["key"]
        with self.lock:
            for unit, label in body["labels"].items():
                if label not in ("pass", "fail"):
                    continue
                self.store.save_label(run_id, version_id, case_id, int(sample_idx), unit, label,
                                      self.labeler, note=body.get("note", ""))
        return {"ok": True, "already_done": self.done_count()}

    def calibrate(self) -> list[dict[str, Any]]:
        out = []
        for t in unit_triples(self.store, self.check):
            rec = run_calibration(self.store, t.key, t.judge_model, t.prompt_hash)
            out.append({"unit": rec.unit_key, "n": rec.n, "agreement": rec.agreement,
                        "kappa": rec.kappa, "sensitivity": rec.sensitivity,
                        "specificity": rec.specificity, "approved": rec.approved})
        return out


def _questions(store: Store, run_id: str, check: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for row in store.score_rows(check=check, run_id=run_id):
        for sub in row.score.sub_scores:
            if isinstance(sub, dict) and sub.get("unit") and isinstance(sub.get("question"), str):
                out.setdefault(str(sub["unit"]), sub["question"])
    return out


def _case_view(case: Case) -> dict[str, Any]:
    """What the labeler needs from a case, and nothing about the judge."""
    source = case.source or {}
    hinted = set(source.get("hinted_atom_ids") or [])
    atoms = [{"id": a.get("id"), "text": a.get("text"), "keywords": a.get("keywords") or [],
              "hinted": a.get("mode") == "hinted" or a.get("id") in hinted}
             for a in source.get("atoms") or [] if isinstance(a, dict)]
    facts = [{"id": f.get("id"), "text": f.get("text") or f"{f.get('subject')}: {f.get('value')}"}
             for f in source.get("facts") or [] if isinstance(f, dict)]
    return {"id": case.id, "input": case.input, "strata": case.strata, "atoms": atoms,
            "facts": facts, "docs": (case.fixtures or {}).get("docs", [])}


def make_handler(session: LabelSession) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args: Any) -> None:  # keep the terminal quiet
            return

        def _send(self, status: int, body: bytes, kind: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: int, data: Any) -> None:
            self._send(status, json.dumps(data, default=str).encode(), "application/json")

        def do_GET(self) -> None:  # noqa: N802
            if self.path in ("/", "/index.html"):
                self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
            elif self.path == "/api/session":
                self._json(200, session.state())
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length") or 0)
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
                if self.path == "/api/label":
                    self._json(200, session.save(body))
                elif self.path == "/api/skip":
                    self._json(200, session.skip(body))
                elif self.path == "/api/calibrate":
                    self._json(200, {"results": session.calibrate()})
                else:
                    self._json(404, {"error": "not found"})
            except Exception as exc:  # report to the page instead of dropping the connection
                self._json(400, {"error": f"{type(exc).__name__}: {exc}"})

    return Handler


def serve(session: LabelSession, host: str = "127.0.0.1", port: int = 8765) -> ThreadingHTTPServer:
    session.load()
    return ThreadingHTTPServer((host, port), make_handler(session))


def export_blind(session: LabelSession) -> tuple[dict[str, Any], dict[str, list]]:
    """The samples still to label, for a page hosted somewhere else (a phone).

    Returns the page data and a key map. The page data carries opaque sample ids
    (s01, s02, ...) instead of the run, version and sample index, so nothing on
    the page says which prompt version wrote a ticket. The key map, kept locally,
    turns those ids back into store keys when the labels come back."""
    session.load()
    samples, keys = [], {}
    for n, smp in enumerate(session.samples, start=1):
        sid = f"s{n:02d}"
        keys[sid] = smp["key"]
        samples.append({"id": sid, "case": smp["case"], "output": smp["output"],
                        "units": smp["units"]})
    data = {"check": session.check, "criteria": session.criteria, "target": len(samples),
            "seen_cases": session.seen_cases(), "samples": samples}
    return data, keys


def import_labels(store: Store, keys: dict[str, list], labels: dict[str, dict[str, Any]],
                  labeler: str) -> dict[str, int]:
    """Save labels from an exported page. `labels` maps a sample id to
    {"labels": {unit: "pass"|"fail"}, "note": str} or {"skipped": true}."""
    counts = {"saved": 0, "skipped": 0, "unknown": 0}
    for sid, body in labels.items():
        key = keys.get(sid)
        if key is None:
            counts["unknown"] += 1
            continue
        given = {u: v for u, v in (body.get("labels") or {}).items() if v in ("pass", "fail")}
        if not given:
            counts["skipped"] += 1
            continue
        run_id, version_id, case_id, sample_idx = key
        for unit, label in given.items():
            store.save_label(run_id, version_id, case_id, int(sample_idx), unit, label, labeler,
                             note=str(body.get("note") or ""))
        counts["saved"] += 1
    return counts


def copy_labels(store: Store, from_run: str, to_run: str, labeler: str) -> dict[str, int]:
    """Carry a labeler's labels to a re-scored run of the same outputs, so a new
    judge prompt can be calibrated without relabeling. A label is copied only when
    the target run holds the identical output for that sample."""
    counts = {"copied": 0, "different_output": 0, "missing": 0}
    for lab in store.labels_for(run_id=from_run, labeler=labeler):
        src = store.get_result(from_run, lab.version_id, lab.case_id, lab.sample_idx)
        dst = store.get_result(to_run, lab.version_id, lab.case_id, lab.sample_idx)
        if src is None or dst is None:
            counts["missing"] += 1
        elif src.output != dst.output:
            counts["different_output"] += 1
        else:
            store.save_label(to_run, lab.version_id, lab.case_id, lab.sample_idx, lab.unit_key,
                             lab.label, labeler, note=lab.note)
            counts["copied"] += 1
    return counts
