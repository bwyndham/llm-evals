"""Case loading, writing and the case-set hash (CASE-1, CASE-3).

Cases live as JSONL, one `Case` per line. `write_cases` writes the canonical
form (sorted keys, compact separators, sorted by id) so that the same set of
cases always produces the same bytes, and `case_set_hash` hashes that form.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path

from pydantic import ValidationError

from llm_evals.types import Case


class CaseLoadError(ValueError):
    """A case file could not be loaded. Carries the file and 1-based line number."""

    def __init__(self, path: Path, line: int | None, message: str) -> None:
        self.path = Path(path)
        self.line = line
        self.message = message
        where = f"{self.path}:{line}" if line is not None else str(self.path)
        super().__init__(f"{where}: {message}")


def canonical_json(case: Case) -> str:
    """One case as canonical JSON: all fields, sorted keys, no extra whitespace."""
    return json.dumps(
        case.model_dump(mode="json"),
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def load_cases(path: Path) -> list[Case]:
    """Read a JSONL case file. Blank lines are skipped; ids must be unique (CASE-1)."""
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise CaseLoadError(path, None, f"cannot read case file: {exc}") from exc

    cases: list[Case] = []
    first_seen: dict[str, int] = {}
    for lineno, raw in enumerate(text.splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise CaseLoadError(path, lineno, f"invalid JSON: {exc.msg}") from exc
        if not isinstance(data, dict):
            raise CaseLoadError(path, lineno, "each line must be a JSON object")
        try:
            case = Case.model_validate(data)
        except ValidationError as exc:
            raise CaseLoadError(path, lineno, f"invalid case: {exc}") from exc
        if case.id in first_seen:
            raise CaseLoadError(
                path,
                lineno,
                f"duplicate case id {case.id!r} (first seen on line {first_seen[case.id]})",
            )
        first_seen[case.id] = lineno
        cases.append(case)
    return cases


def _sorted_unique(cases: Iterable[Case]) -> list[Case]:
    ordered = sorted(cases, key=lambda c: c.id)
    for prev, cur in zip(ordered, ordered[1:]):
        if prev.id == cur.id:
            raise ValueError(f"duplicate case id {cur.id!r}")
    return ordered


def write_cases(path: Path, cases: Iterable[Case]) -> Path:
    """Write cases as canonical JSONL, sorted by id. Returns the path written."""
    path = Path(path)
    ordered = _sorted_unique(cases)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "".join(canonical_json(c) + "\n" for c in ordered)
    path.write_text(body, encoding="utf-8")
    return path


def case_set_hash(cases: Iterable[Case]) -> str:
    """Stable sha256 hex of the canonical cases sorted by id (CASE-3). Order-independent."""
    ordered = _sorted_unique(cases)
    digest = hashlib.sha256()
    for case in ordered:
        digest.update(canonical_json(case).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()
