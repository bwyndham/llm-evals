"""SQLite results store: runs, results (the RUN-5 cache), scores, baselines,
and (schema v2) judge calibration: run cases, human labels and calibrations.

One connection guarded by a lock, so the concurrency pool's threads can write
through a single Store. `Store(":memory:")` works for tests.

Schema history:
- v1: runs, results, scores, baselines.
- v2: adds run_cases, labels and calibrations (CAL-1, CAL-2, CAL-4). Opening a
  v1 file creates the new tables and bumps the version; nothing else changes.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from llm_evals.types import Case, ErrorKind, Result, Score, VersionManifest

SCHEMA_VERSION = 2

_SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    suite TEXT NOT NULL,
    kind TEXT NOT NULL,
    started_at TEXT NOT NULL,
    config_hash TEXT,
    case_set_hash TEXT,
    baseline_version_id TEXT,
    candidate_version_id TEXT,
    status TEXT NOT NULL,
    decision TEXT,
    report_path TEXT,
    meta TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS results (
    run_id TEXT NOT NULL,
    version_id TEXT NOT NULL,
    case_id TEXT NOT NULL,
    sample_idx INTEGER NOT NULL,
    error_kind TEXT NOT NULL,
    result_json TEXT NOT NULL,
    PRIMARY KEY (run_id, version_id, case_id, sample_idx)
);
CREATE TABLE IF NOT EXISTS scores (
    run_id TEXT NOT NULL,
    version_id TEXT NOT NULL,
    case_id TEXT NOT NULL,
    sample_idx INTEGER NOT NULL,
    "check" TEXT NOT NULL,
    score_json TEXT NOT NULL,
    PRIMARY KEY (run_id, version_id, case_id, sample_idx, "check")
);
CREATE TABLE IF NOT EXISTS baselines (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    suite TEXT NOT NULL,
    version_id TEXT NOT NULL,
    manifest_path TEXT,
    manifest_json TEXT NOT NULL,
    promoted_at TEXT NOT NULL,
    run_id TEXT
);
CREATE INDEX IF NOT EXISTS baselines_suite ON baselines (suite, seq);
"""

# v2: judge calibration. Applied to fresh files and as the v1 -> v2 migration.
_SCHEMA_V2 = """
CREATE TABLE IF NOT EXISTS run_cases (
    run_id TEXT NOT NULL,
    case_id TEXT NOT NULL,
    case_json TEXT NOT NULL,
    PRIMARY KEY (run_id, case_id)
);
CREATE TABLE IF NOT EXISTS labels (
    run_id TEXT NOT NULL,
    version_id TEXT NOT NULL,
    case_id TEXT NOT NULL,
    sample_idx INTEGER NOT NULL,
    unit_key TEXT NOT NULL,
    label TEXT NOT NULL CHECK (label IN ('pass', 'fail')),
    labeler TEXT NOT NULL,
    note TEXT NOT NULL DEFAULT '',
    labeled_at TEXT NOT NULL,
    UNIQUE (run_id, version_id, case_id, sample_idx, unit_key, labeler)
);
CREATE INDEX IF NOT EXISTS labels_unit ON labels (unit_key);
CREATE TABLE IF NOT EXISTS calibrations (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    unit_key TEXT NOT NULL,
    judge_model TEXT NOT NULL,
    prompt_hash TEXT NOT NULL,
    n INTEGER NOT NULL,
    agreement REAL,
    kappa REAL,
    sensitivity REAL,
    specificity REAL,
    prevalence REAL,
    tp INTEGER NOT NULL DEFAULT 0,
    fp INTEGER NOT NULL DEFAULT 0,
    fn INTEGER NOT NULL DEFAULT 0,
    tn INTEGER NOT NULL DEFAULT 0,
    min_n INTEGER,
    min_kappa REAL,
    approved INTEGER NOT NULL,
    label_set TEXT NOT NULL DEFAULT '{}',
    computed_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS calibrations_triple
    ON calibrations (unit_key, judge_model, prompt_hash, seq);
"""

# version -> SQL that upgrades a file from that version to the next one.
_MIGRATIONS: dict[int, str] = {1: _SCHEMA_V2}

RUN_COLUMNS = (
    "id",
    "suite",
    "kind",
    "started_at",
    "config_hash",
    "case_set_hash",
    "baseline_version_id",
    "candidate_version_id",
    "status",
    "decision",
    "report_path",
    "meta",
)
_UPDATABLE_RUN_FIELDS = set(RUN_COLUMNS) - {"id"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class _Subscriptable:
    """Records also allow `record["field"]` access."""

    def __getitem__(self, key: str) -> Any:
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(key) from None


@dataclass
class RunRecord(_Subscriptable):
    id: str
    suite: str
    kind: str
    started_at: str
    config_hash: str | None = None
    case_set_hash: str | None = None
    baseline_version_id: str | None = None
    candidate_version_id: str | None = None
    status: str = "running"
    decision: str | None = None
    report_path: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class LabelRecord(_Subscriptable):
    """One human verdict on one gating unit of one sample (CAL-1)."""

    run_id: str
    version_id: str
    case_id: str
    sample_idx: int
    unit_key: str
    label: str  # "pass" | "fail"
    labeler: str
    note: str = ""
    labeled_at: str = ""


@dataclass
class UnitVerdict(_Subscriptable):
    """A judge's verdict on one gating unit of one sample, read from stored scores."""

    run_id: str
    version_id: str
    case_id: str
    sample_idx: int
    check: str
    unit_key: str
    passed: bool
    judge_model: str | None
    prompt_hash: str | None


@dataclass
class ScoreRow(_Subscriptable):
    run_id: str
    version_id: str
    case_id: str
    sample_idx: int
    score: Score


@dataclass
class CalibrationRecord(_Subscriptable):
    """Human-vs-judge agreement for one (unit, judge model, prompt hash) (CAL-2).

    Positive class = FAIL: sensitivity is the share of human-labeled failures the
    judge also failed; specificity is the share of human-labeled passes the judge
    also passed; prevalence is the human failure rate.
    """

    unit_key: str
    judge_model: str
    prompt_hash: str
    n: int
    agreement: float | None
    kappa: float | None
    sensitivity: float | None
    specificity: float | None
    prevalence: float | None
    approved: bool
    label_set: dict[str, Any] = field(default_factory=dict)  # {"run_ids": [...], "count": n}
    computed_at: str = ""
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0
    min_n: int | None = None
    min_kappa: float | None = None
    seq: int | None = None  # set when read back from the store


_CALIBRATION_COLUMNS = (
    "seq", "unit_key", "judge_model", "prompt_hash", "n", "agreement", "kappa",
    "sensitivity", "specificity", "prevalence", "tp", "fp", "fn", "tn", "min_n",
    "min_kappa", "approved", "label_set", "computed_at",
)


def _calibration_from_row(row: sqlite3.Row) -> CalibrationRecord:
    data = dict(row)
    data["approved"] = bool(data["approved"])
    data["label_set"] = json.loads(data["label_set"] or "{}")
    return CalibrationRecord(**data)


def unit_entries(score: Score) -> list[tuple[str, bool, str | None, str | None]]:
    """The gating-unit sub_scores of a Score as (unit_key, passed, judge_model, prompt_hash).

    Per the judges contract each unit entry is `{"unit": key, "passed": bool, ...}`.
    A unit's judge model and prompt hash come from the entry's own `judge_model` /
    `prompt_hash` keys when present, falling back to `Score.judge_model` /
    `Score.prompt_hash` (one-unit checks such as faithfulness only set the latter).
    """
    out = []
    for sub in score.sub_scores:
        key = sub.get("unit") if isinstance(sub, dict) else None
        if not key or "passed" not in sub:
            continue
        out.append((
            str(key),
            bool(sub["passed"]),
            sub.get("judge_model") or score.judge_model,
            sub.get("prompt_hash") or score.prompt_hash,
        ))
    return out


@dataclass
class BaselineRecord(_Subscriptable):
    suite: str
    version_id: str
    manifest: VersionManifest
    manifest_path: str | None
    promoted_at: str
    run_id: str | None


class StoreError(RuntimeError):
    pass


class Store:
    def __init__(self, path: Path | str) -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            if self.path != ":memory:":
                self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.executescript(_SCHEMA)
            row = self._conn.execute("SELECT version FROM schema_version").fetchone()
            if row is None:
                self._conn.executescript(_SCHEMA_V2)
                self._conn.execute("INSERT INTO schema_version (version) VALUES (?)", (SCHEMA_VERSION,))
            elif row["version"] > SCHEMA_VERSION:
                raise StoreError(
                    f"{self.path} has schema version {row['version']}, expected {SCHEMA_VERSION}"
                )
            else:
                version = row["version"]
                while version < SCHEMA_VERSION:
                    self._conn.executescript(_MIGRATIONS[version])
                    version += 1
                self._conn.execute("UPDATE schema_version SET version = ?", (version,))
            self._conn.commit()

    # --- lifecycle ---------------------------------------------------------

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> Store:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _write(self, sql: str, params: tuple[Any, ...] = ()) -> sqlite3.Cursor:
        with self._lock:
            cur = self._conn.execute(sql, params)
            self._conn.commit()
            return cur

    def _read(self, sql: str, params: tuple[Any, ...] = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(sql, params).fetchall()

    # --- runs --------------------------------------------------------------

    def create_run(
        self,
        *,
        suite: str,
        kind: str,
        config_hash: str | None = None,
        case_set_hash: str | None = None,
        baseline_version_id: str | None = None,
        candidate_version_id: str | None = None,
        status: str = "running",
        meta: dict[str, Any] | None = None,
        run_id: str | None = None,
        started_at: str | None = None,
    ) -> str:
        run_id = run_id or uuid.uuid4().hex
        self._write(
            "INSERT INTO runs (id, suite, kind, started_at, config_hash, case_set_hash, "
            "baseline_version_id, candidate_version_id, status, meta) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                run_id,
                suite,
                kind,
                started_at or _now(),
                config_hash,
                case_set_hash,
                baseline_version_id,
                candidate_version_id,
                status,
                json.dumps(meta or {}, sort_keys=True),
            ),
        )
        return run_id

    def update_run(self, run_id: str, **fields: Any) -> None:
        unknown = set(fields) - _UPDATABLE_RUN_FIELDS
        if unknown:
            raise ValueError(f"unknown run fields: {sorted(unknown)}")
        if not fields:
            return
        if "meta" in fields:
            fields["meta"] = json.dumps(fields["meta"] or {}, sort_keys=True)
        if "decision" in fields and fields["decision"] is not None:
            fields["decision"] = getattr(fields["decision"], "value", fields["decision"])
        if "report_path" in fields and fields["report_path"] is not None:
            fields["report_path"] = str(fields["report_path"])
        assignments = ", ".join(f"{name} = ?" for name in fields)
        cur = self._write(
            f"UPDATE runs SET {assignments} WHERE id = ?", (*fields.values(), run_id)
        )
        if cur.rowcount == 0:
            raise KeyError(f"no run {run_id!r}")

    def get_run(self, run_id: str) -> RunRecord | None:
        rows = self._read(f"SELECT {', '.join(RUN_COLUMNS)} FROM runs WHERE id = ?", (run_id,))
        if not rows:
            return None
        data = dict(rows[0])
        data["meta"] = json.loads(data["meta"] or "{}")
        return RunRecord(**data)

    # --- results (RUN-5 cache) ---------------------------------------------

    def save_result(self, result: Result) -> None:
        """Insert or replace the result for (run, version, case, sample)."""
        self._write(
            "INSERT OR REPLACE INTO results "
            "(run_id, version_id, case_id, sample_idx, error_kind, result_json) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                result.run_id,
                result.version_id,
                result.case_id,
                result.sample_idx,
                result.error_kind.value,
                result.model_dump_json(),
            ),
        )

    def get_result(
        self,
        run_id: str,
        version_id: str,
        case_id: str,
        sample_idx: int,
        *,
        include_infra: bool = True,
    ) -> Result | None:
        """The cached result, or None. With include_infra=False an INFRA result
        counts as a miss, so a resumed run retries it."""
        rows = self._read(
            "SELECT error_kind, result_json FROM results "
            "WHERE run_id = ? AND version_id = ? AND case_id = ? AND sample_idx = ?",
            (run_id, version_id, case_id, sample_idx),
        )
        if not rows:
            return None
        if not include_infra and rows[0]["error_kind"] == ErrorKind.INFRA.value:
            return None
        return Result.model_validate_json(rows[0]["result_json"])

    def results_for(self, run_id: str, version_id: str) -> list[Result]:
        rows = self._read(
            "SELECT result_json FROM results WHERE run_id = ? AND version_id = ? "
            "ORDER BY case_id, sample_idx",
            (run_id, version_id),
        )
        return [Result.model_validate_json(r["result_json"]) for r in rows]

    # --- scores ------------------------------------------------------------

    def save_score(
        self, run_id: str, version_id: str, case_id: str, sample_idx: int, score: Score
    ) -> None:
        self._write(
            'INSERT OR REPLACE INTO scores (run_id, version_id, case_id, sample_idx, "check", '
            "score_json) VALUES (?, ?, ?, ?, ?, ?)",
            (run_id, version_id, case_id, sample_idx, score.check, score.model_dump_json()),
        )

    def scores_for(self, run_id: str, version_id: str) -> list[tuple[str, int, Score]]:
        rows = self._read(
            "SELECT case_id, sample_idx, score_json FROM scores "
            'WHERE run_id = ? AND version_id = ? ORDER BY case_id, sample_idx, "check"',
            (run_id, version_id),
        )
        return [
            (r["case_id"], r["sample_idx"], Score.model_validate_json(r["score_json"]))
            for r in rows
        ]

    # --- baselines (REP-3: history kept) -----------------------------------

    def promote(
        self,
        suite: str,
        manifest: VersionManifest,
        manifest_path: Path | str | None,
        run_id: str | None,
    ) -> BaselineRecord:
        if not manifest.id:
            raise ValueError("manifest has no id; load it with load_manifest")
        promoted_at = _now()
        self._write(
            "INSERT INTO baselines (suite, version_id, manifest_path, manifest_json, "
            "promoted_at, run_id) VALUES (?, ?, ?, ?, ?, ?)",
            (
                suite,
                manifest.id,
                str(manifest_path) if manifest_path is not None else None,
                manifest.model_dump_json(),
                promoted_at,
                run_id,
            ),
        )
        return BaselineRecord(
            suite=suite,
            version_id=manifest.id,
            manifest=manifest,
            manifest_path=str(manifest_path) if manifest_path is not None else None,
            promoted_at=promoted_at,
            run_id=run_id,
        )

    def _baselines(self, suite: str, limit: int | None) -> list[BaselineRecord]:
        sql = (
            "SELECT suite, version_id, manifest_path, manifest_json, promoted_at, run_id "
            "FROM baselines WHERE suite = ? ORDER BY seq DESC"
        )
        params: tuple[Any, ...] = (suite,)
        if limit is not None:
            sql += " LIMIT ?"
            params = (suite, limit)
        return [
            BaselineRecord(
                suite=r["suite"],
                version_id=r["version_id"],
                manifest=VersionManifest.model_validate_json(r["manifest_json"]),
                manifest_path=r["manifest_path"],
                promoted_at=r["promoted_at"],
                run_id=r["run_id"],
            )
            for r in self._read(sql, params)
        ]

    def current_baseline(self, suite: str) -> BaselineRecord | None:
        found = self._baselines(suite, 1)
        return found[0] if found else None

    def baseline_history(self, suite: str) -> list[BaselineRecord]:
        """Every promotion for the suite, newest first."""
        return self._baselines(suite, None)

    # --- calibration (schema v2): run cases, labels, calibrations ---------

    def save_run_cases(self, run_id: str, cases: list[Case]) -> None:
        """Record the cases a run used, so labeling can show inputs and source facts."""
        with self._lock:
            self._conn.executemany(
                "INSERT OR REPLACE INTO run_cases (run_id, case_id, case_json) VALUES (?, ?, ?)",
                [(run_id, c.id, c.model_dump_json()) for c in cases],
            )
            self._conn.commit()

    def run_cases(self, run_id: str) -> dict[str, Case]:
        rows = self._read(
            "SELECT case_id, case_json FROM run_cases WHERE run_id = ? ORDER BY case_id", (run_id,)
        )
        return {r["case_id"]: Case.model_validate_json(r["case_json"]) for r in rows}

    def score_rows(self, *, check: str | None = None, run_id: str | None = None) -> list[ScoreRow]:
        """Stored scores across runs, optionally for one check and/or one run."""
        sql = "SELECT run_id, version_id, case_id, sample_idx, score_json FROM scores"
        clauses, params = [], []
        if check is not None:
            clauses.append('"check" = ?')
            params.append(check)
        if run_id is not None:
            clauses.append("run_id = ?")
            params.append(run_id)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += ' ORDER BY run_id, version_id, case_id, sample_idx, "check"'
        return [
            ScoreRow(r["run_id"], r["version_id"], r["case_id"], r["sample_idx"],
                     Score.model_validate_json(r["score_json"]))
            for r in self._read(sql, tuple(params))
        ]

    def unit_verdicts(self, unit_key: str, *, check: str | None = None) -> list[UnitVerdict]:
        """Every stored judge verdict for one gating unit, across all runs."""
        out = []
        for row in self.score_rows(check=check):
            for key, passed, model, phash in unit_entries(row.score):
                if key == unit_key:
                    out.append(UnitVerdict(row.run_id, row.version_id, row.case_id,
                                           row.sample_idx, row.score.check, key, passed,
                                           model, phash))
        return out

    def save_label(
        self,
        run_id: str,
        version_id: str,
        case_id: str,
        sample_idx: int,
        unit_key: str,
        label: str,
        labeler: str,
        note: str = "",
        labeled_at: str | None = None,
    ) -> LabelRecord:
        """Insert a human label, replacing this labeler's earlier label for the same unit."""
        if label not in ("pass", "fail"):
            raise ValueError(f"label must be 'pass' or 'fail', got {label!r}")
        if not labeler:
            raise ValueError("labeler is required")
        record = LabelRecord(run_id, version_id, case_id, sample_idx, unit_key, label,
                             labeler, note or "", labeled_at or _now())
        self._write(
            "INSERT INTO labels (run_id, version_id, case_id, sample_idx, unit_key, label, "
            "labeler, note, labeled_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT (run_id, version_id, case_id, sample_idx, unit_key, labeler) "
            "DO UPDATE SET label = excluded.label, note = excluded.note, "
            "labeled_at = excluded.labeled_at",
            (record.run_id, record.version_id, record.case_id, record.sample_idx,
             record.unit_key, record.label, record.labeler, record.note, record.labeled_at),
        )
        return record

    def labels_for(
        self,
        unit_key: str | None = None,
        run_id: str | None = None,
        *,
        labeler: str | None = None,
    ) -> list[LabelRecord]:
        sql = ("SELECT run_id, version_id, case_id, sample_idx, unit_key, label, labeler, "
               "note, labeled_at FROM labels")
        clauses, params = [], []
        for column, value in (("unit_key", unit_key), ("run_id", run_id), ("labeler", labeler)):
            if value is not None:
                clauses.append(f"{column} = ?")
                params.append(value)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY run_id, version_id, case_id, sample_idx, unit_key, labeler"
        return [LabelRecord(**dict(r)) for r in self._read(sql, tuple(params))]

    def save_calibration(self, record: CalibrationRecord) -> CalibrationRecord:
        """Append a calibration (history is kept); returns it with `seq` and `computed_at` set."""
        computed_at = record.computed_at or _now()
        cur = self._write(
            "INSERT INTO calibrations (unit_key, judge_model, prompt_hash, n, agreement, kappa, "
            "sensitivity, specificity, prevalence, tp, fp, fn, tn, min_n, min_kappa, approved, "
            "label_set, computed_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (record.unit_key, record.judge_model, record.prompt_hash, record.n,
             record.agreement, record.kappa, record.sensitivity, record.specificity,
             record.prevalence, record.tp, record.fp, record.fn, record.tn, record.min_n,
             record.min_kappa, int(bool(record.approved)),
             json.dumps(record.label_set or {}, sort_keys=True), computed_at),
        )
        record.seq = cur.lastrowid
        record.computed_at = computed_at
        return record

    def latest_calibration(
        self, unit_key: str, judge_model: str, prompt_hash: str
    ) -> CalibrationRecord | None:
        rows = self._read(
            f"SELECT {', '.join(_CALIBRATION_COLUMNS)} FROM calibrations "
            "WHERE unit_key = ? AND judge_model = ? AND prompt_hash = ? ORDER BY seq DESC LIMIT 1",
            (unit_key, judge_model, prompt_hash),
        )
        return _calibration_from_row(rows[0]) if rows else None

    def approved_calibration(
        self,
        unit_key: str,
        judge_model: str,
        prompt_hash: str,
        *,
        min_n: int = 50,
        min_kappa: float = 0.6,
    ) -> bool:
        """True iff the latest calibration for exactly this triple is approved and meets
        both bars (n >= min_n, kappa >= min_kappa). A changed judge model or prompt hash
        has no calibration, so the unit is report-only until recalibrated (CAL-4)."""
        latest = self.latest_calibration(unit_key, judge_model, prompt_hash)
        return (
            latest is not None
            and latest.approved
            and latest.n >= min_n
            and latest.kappa is not None
            and latest.kappa >= min_kappa
        )

    def calibration_history(self, unit_key: str) -> list[CalibrationRecord]:
        """Every calibration of a unit (any judge model / prompt hash), newest first."""
        rows = self._read(
            f"SELECT {', '.join(_CALIBRATION_COLUMNS)} FROM calibrations "
            "WHERE unit_key = ? ORDER BY seq DESC",
            (unit_key,),
        )
        return [_calibration_from_row(r) for r in rows]

    def latest_calibrations(self) -> list[CalibrationRecord]:
        """The newest calibration of every (unit, judge model, prompt hash), by unit."""
        rows = self._read(
            f"SELECT {', '.join(_CALIBRATION_COLUMNS)} FROM calibrations c "
            "WHERE seq = (SELECT MAX(seq) FROM calibrations d WHERE d.unit_key = c.unit_key "
            "AND d.judge_model = c.judge_model AND d.prompt_hash = c.prompt_hash) "
            "ORDER BY unit_key, judge_model, prompt_hash"
        )
        return [_calibration_from_row(r) for r in rows]
