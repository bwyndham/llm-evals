"""Checks for the research agent (CHK-9).

The agent answers a question from a small per-case corpus (`case.fixtures.docs`)
with tool calls recorded in `result.trace.tool_calls`. Its output is

    {"status": "answered" | "not_found", "answer": str,
     "claims": [{"text": str, "citations": [doc_id]}], "conflicts": [str]}

Checks registered here:
- `citations_exist` (structural, critical by default): every cited doc id was
  returned or fetched by a tool call, and exists in the corpus at all.
- `answer_match` (reference): the behavior in `case.expected` happened
  (answer / abstain / flag_conflict).
- `follows_corpus` (metamorphic): on `change_fact` cases the answer follows the
  changed corpus, not the model's prior (the original value).
- `uncited_claims` (structural): at most `max_share` of claims lack citations.
- `claim_support` (faithfulness, LLM judge): each claim is supported by the text
  of the documents it cites.

Answer matching. `llm_evals.cases.corpus.answer_matches` (owned by the
research-data module) is the suite's definition of a correct answer, so it is
preferred whenever it imports and runs. This module carries its own small
matcher, `local_answer_matches`, used when the corpus helper is missing or
raises. The local matcher: NFKC + casefold + accent stripping; punctuation to
spaces; thousands separators removed ("12,500" -> "12500"); leading articles
dropped. A candidate (the expected answer or an alias) matches when its
normalized form equals the normalized output or appears in it on token
boundaries. When the candidate starts with a number ("12,500 people",
"$4.2 million", "45%"), it also matches when any number in the output has the
same value, with scale words (thousand, million, billion, trillion, k, bn, mn)
applied and units ignored.
"""

from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from llm_evals.checks._base import BaseCheck, no_output_reason, truncate
from llm_evals.registry import register_check
from llm_evals.runners.errors import FatalRunError, classify_exception
from llm_evals.types import Case, CheckContext, CheckKind, ErrorKind, GatingUnit, Result, Score

STATUSES = ("answered", "not_found")
BEHAVIORS = ("answer", "abstain", "flag_conflict")

# The research-agent suite directory, used when neither the `suite_dir` param
# nor a CheckContext says where the judge templates live.
_DEFAULT_SUITE_DIR = Path(__file__).resolve().parents[3] / "suites" / "research-agent"


# --- Output parsing -----------------------------------------------------------


class MalformedOutput(ValueError):
    """The agent output doesn't have the contract's shape."""


def _status(output: dict[str, Any]) -> str:
    status = output.get("status")
    if status not in STATUSES:
        raise MalformedOutput(f"status must be one of {', '.join(STATUSES)}, got {status!r}")
    return str(status)


def _answer(output: dict[str, Any]) -> str:
    answer = output.get("answer")
    if answer is None:
        return ""
    if not isinstance(answer, str):
        raise MalformedOutput(f"answer must be a string, got {type(answer).__name__}")
    return answer


def _claims(output: dict[str, Any]) -> list[dict[str, Any]]:
    """The claims as [{"text": str, "citations": [str]}]; missing means none."""
    raw = output.get("claims")
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise MalformedOutput(f"claims must be a list, got {type(raw).__name__}")
    claims: list[dict[str, Any]] = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            raise MalformedOutput(f"claims[{i}] must be an object, got {type(item).__name__}")
        text = item.get("text")
        if text is not None and not isinstance(text, str):
            raise MalformedOutput(f"claims[{i}].text must be a string, got {type(text).__name__}")
        cites = item.get("citations")
        if cites is None:
            cites = []
        if not isinstance(cites, list) or not all(isinstance(c, str) for c in cites):
            raise MalformedOutput(f"claims[{i}].citations must be a list of doc id strings, got {cites!r}")
        claims.append({"text": text or "", "citations": [c.strip() for c in cites if c.strip()]})
    return claims


def _conflicts(output: dict[str, Any]) -> list[str]:
    raw = output.get("conflicts")
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise MalformedOutput(f"conflicts must be a list, got {type(raw).__name__}")
    return [str(c) for c in raw if str(c).strip()]


def corpus_docs(case: Case) -> dict[str, dict[str, Any]] | None:
    """doc id -> doc from `case.fixtures.docs`, or None if the case has no corpus."""
    fixtures = case.fixtures or {}
    docs = fixtures.get("docs")
    if not isinstance(docs, list):
        return None
    return {str(d["id"]): d for d in docs if isinstance(d, dict) and "id" in d}


def retrieved_doc_ids(result: Result) -> set[str]:
    """Every doc id a tool call returned or fetched."""
    seen: set[str] = set()
    for call in result.trace.tool_calls:
        if not isinstance(call, dict):
            continue
        ids = call.get("doc_ids")
        if ids is None and call.get("name") == "fetch":
            # Older traces may omit doc_ids; a fetch still names its document.
            doc_id = (call.get("input") or {}).get("doc_id")
            ids = [doc_id] if doc_id else []
        for doc_id in ids or []:
            if doc_id is not None:
                seen.add(str(doc_id))
    return seen


# --- Answer matching ----------------------------------------------------------

_ARTICLES = {"a", "an", "the"}
# Thousands separators between digits: comma, apostrophe, no-break/thin spaces.
_DIGIT_SEPARATOR = re.compile("(?<=\\d)[,'\u00a0\u2009\u202f](?=\\d{3}\\b)")
_NUMBER = re.compile(
    r"(?<![\w.])(?P<num>[-+]?\d+(?:\.\d+)?)(?:\s*(?P<scale>thousand|million|billion|trillion|k|mn|bn)\b)?",
    re.IGNORECASE,
)
_LEADING_NUMBER = re.compile(r"^\s*[-+]?\s*[$€£¥]?\s*\d")
_SCALES = {"thousand": 1e3, "k": 1e3, "million": 1e6, "mn": 1e6, "billion": 1e9, "bn": 1e9, "trillion": 1e12}


def _strip_separators(text: str) -> str:
    """'12,500' / '12 500' (no-break space) -> '12500'. Runs before NFKC folds the spaces."""
    return _DIGIT_SEPARATOR.sub("", str(text))


def normalize_answer(text: str) -> str:
    """Lowercase, accent-free, punctuation-free text with digit separators removed."""
    text = unicodedata.normalize("NFKC", _strip_separators(text)).casefold()
    text = "".join(ch for ch in unicodedata.normalize("NFKD", text) if not unicodedata.combining(ch))
    text = re.sub(r"(?<=\d)\.(?=\d)", "\x00", text)  # keep decimal points
    text = re.sub(r"[^\w\x00]+", " ", text).replace("\x00", ".").replace("_", " ")
    tokens = text.split()
    while tokens and tokens[0] in _ARTICLES:
        tokens = tokens[1:]
    return " ".join(tokens)


def extract_numbers(text: str) -> list[float]:
    """Numeric values in `text`, with thousands separators and scale words applied."""
    values: list[float] = []
    for match in _NUMBER.finditer(unicodedata.normalize("NFKC", _strip_separators(text))):
        value = float(match.group("num"))
        if match.group("scale"):
            value *= _SCALES[match.group("scale").lower()]
        values.append(value)
    return values


def _same_number(a: float, b: float) -> bool:
    return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9)


def _candidate_matches(output_answer: str, candidate: str) -> bool:
    want = normalize_answer(candidate)
    have = normalize_answer(output_answer)
    if not want or not have:
        return False
    if want == have or f" {want} " in f" {have} ":
        return True
    if _LEADING_NUMBER.match(candidate):
        wanted = extract_numbers(candidate)
        if wanted:
            return any(_same_number(wanted[0], got) for got in extract_numbers(output_answer))
    return False


def local_answer_matches(output_answer: str, expected: dict[str, Any]) -> bool:
    """This module's fallback matcher (see the module docstring)."""
    candidates = [expected.get("answer"), *(expected.get("aliases") or [])]
    return any(_candidate_matches(output_answer, str(c)) for c in candidates if c not in (None, ""))


def _corpus_matcher() -> Callable[[str, dict[str, Any]], bool] | None:
    """`llm_evals.cases.corpus.answer_matches` if it imports, else None."""
    try:
        from llm_evals.cases.corpus import answer_matches
    except Exception:  # noqa: BLE001 - the corpus module may not exist yet
        return None
    return answer_matches


def answer_matches(output_answer: str, expected: dict[str, Any]) -> bool:
    """Prefer the corpus helper; fall back to the local matcher if it's missing or raises."""
    matcher = _corpus_matcher()
    if matcher is not None:
        try:
            return bool(matcher(output_answer, expected))
        except Exception:  # noqa: BLE001
            pass
    return local_answer_matches(output_answer, expected)


def _matches_value(output_answer: str, value: str, aliases: list[str] | None = None) -> bool:
    return answer_matches(output_answer, {"answer": value, "aliases": list(aliases or [])})


def _expected(case: Case) -> dict[str, Any]:
    if not isinstance(case.expected, dict):
        raise ValueError("case has no expected block")
    return case.expected


def _perturbation(case: Case) -> dict[str, Any]:
    source = case.source or {}
    pert = source.get("perturbation")
    return pert if isinstance(pert, dict) else {}


# Where a flag_conflict case may name the values that disagree. The research-data
# module owns the case shape; these are the keys this check reads.
_CONFLICT_LIST_KEYS = ("conflicting_values", "values")
_CONFLICT_VALUE_KEYS = (
    "value", "new_value", "conflict_value", "conflicting_value", "old_value", "original_value",
)


def conflicting_values(case: Case) -> list[str]:
    """The distinct values a flag_conflict case's documents disagree on."""
    expected = case.expected or {}
    pert = _perturbation(case)
    found: list[str] = []
    for block in (expected, pert):
        for key in _CONFLICT_LIST_KEYS:
            if isinstance(block.get(key), list):
                found.extend(str(v) for v in block[key] if v not in (None, ""))
    found.extend(str(expected[k]) for k in ("answer", "original_answer") if expected.get(k) not in (None, ""))
    found.extend(
        str(pert[k]) for k in _CONFLICT_VALUE_KEYS
        if pert.get(k) not in (None, "") and not isinstance(pert[k], (list, dict))
    )
    distinct: dict[str, str] = {}
    for value in found:
        distinct.setdefault(normalize_answer(value), value)
    return [v for k, v in distinct.items() if k]


# --- Checks -------------------------------------------------------------------


class ResearchCheck(BaseCheck):
    """BaseCheck that also hands the Result (for its trace) and reports malformed output."""

    def score(self, case: Case, result: Result, ctx: CheckContext) -> Score:
        if not self.applies(case):
            return self.not_applicable_score(case)
        missing = no_output_reason(result)
        if missing is not None:
            return self.fail(missing)
        try:
            return self._score_research(case, result, result.output or {}, ctx)
        except MalformedOutput as exc:
            return self.fail(f"malformed output: {exc}")
        except FatalRunError:  # config problem (e.g. a judge template error): abort the run
            raise
        except Exception as exc:  # noqa: BLE001 - a check must never raise
            return self.fail(f"check error: {type(exc).__name__}: {exc}")

    def _score_research(self, case: Case, result: Result, output: dict[str, Any], ctx: CheckContext) -> Score:
        raise NotImplementedError

    def ok(self, reason: str, **fields: Any) -> Score:
        return Score(check=self.name, kind=self.kind, passed=True, reason=reason, **fields)


class CitationsExist(ResearchCheck):
    """Every cited doc id was retrieved (search result or fetch) and exists in the corpus.

    One sub-score per bad citation:
    `{"claim": i, "doc_id": id, "problem": "not_in_corpus" | "not_retrieved",
      "severity": "high" | "medium", "claim_text": str}`.
    A citation to a document that isn't in the corpus at all is a fabricated source
    ("not_in_corpus", severity high); one that exists but the agent never saw is
    "not_retrieved" (severity medium). `value` = share of citations that are good.
    """

    name = "citations_exist"
    kind: CheckKind = "structural"

    def _score_research(self, case: Case, result: Result, output: dict[str, Any], ctx: CheckContext) -> Score:
        status = _status(output)
        claims = _claims(output)
        if not claims:
            if status == "not_found":
                return self.ok("no claims (status not_found): nothing to cite")
            return self.ok("no claims to check (uncited_claims judges an answer without claims)")

        corpus = corpus_docs(case)
        retrieved = retrieved_doc_ids(result)
        bad: list[dict[str, Any]] = []
        total = 0
        for i, claim in enumerate(claims):
            for doc_id in claim["citations"]:
                total += 1
                if corpus is not None and doc_id not in corpus:
                    problem, severity = "not_in_corpus", "high"
                elif doc_id not in retrieved:
                    problem, severity = "not_retrieved", "medium"
                else:
                    continue
                bad.append({
                    "claim": i,
                    "doc_id": doc_id,
                    "problem": problem,
                    "severity": severity,
                    "claim_text": truncate(claim["text"], 300),
                })

        if total == 0:
            return self.ok("no citations to check (uncited_claims judges uncited claims)")
        value = (total - len(bad)) / total
        if not bad:
            note = "" if corpus is not None else " (case has no corpus; checked retrieval only)"
            return self.ok(f"all {total} citation(s) were retrieved{note}", value=value)
        missing = sorted({b["doc_id"] for b in bad if b["problem"] == "not_in_corpus"})
        unseen = sorted({b["doc_id"] for b in bad if b["problem"] == "not_retrieved"})
        parts = []
        if missing:
            parts.append(f"cited doc(s) not in the corpus: {', '.join(missing)}")
        if unseen:
            parts.append(f"cited doc(s) never retrieved or fetched: {', '.join(unseen)}")
        return self.fail(f"{len(bad)} of {total} citation(s) bad; " + "; ".join(parts), value=value, sub_scores=bad)


class AnswerMatch(ResearchCheck):
    """The expected behavior happened.

    - answer: status answered and the answer matches `expected.answer` or an alias;
    - abstain: status not_found (any answer text is ignored);
    - flag_conflict: `conflicts` is non-empty and the answer does not assert just
      one of the conflicting values (it may name all of them, or none).
    Sub-scores: one `{"behavior", "status", "answer", ...}` summary.
    """

    name = "answer_match"
    kind: CheckKind = "reference"

    def _score_research(self, case: Case, result: Result, output: dict[str, Any], ctx: CheckContext) -> Score:
        expected = _expected(case)
        behavior = expected.get("behavior", "answer")
        if behavior not in BEHAVIORS:
            raise ValueError(f"unknown expected.behavior {behavior!r} (known: {', '.join(BEHAVIORS)})")
        status = _status(output)
        answer = _answer(output)
        summary: dict[str, Any] = {"behavior": behavior, "status": status, "answer": truncate(answer, 300)}

        if behavior == "abstain":
            if status == "not_found":
                return self.ok("abstained as expected", sub_scores=[summary])
            return self.fail("expected the agent to abstain (not_found) but it answered", sub_scores=[summary])

        if behavior == "answer":
            summary["expected"] = expected.get("answer")
            if status != "answered":
                return self.fail(
                    f"expected an answer ({expected.get('answer')!r}) but status is {status}", sub_scores=[summary]
                )
            matched = answer_matches(answer, expected)
            summary["matched"] = matched
            if matched:
                return self.ok(f"answer matches {expected.get('answer')!r}", sub_scores=[summary])
            return self.fail(
                f"answer {truncate(answer, 200)!r} does not match {expected.get('answer')!r}"
                f" (aliases: {expected.get('aliases') or []})",
                sub_scores=[summary],
            )

        # flag_conflict
        conflicts = _conflicts(output)
        values = conflicting_values(case)
        asserted = [v for v in values if _matches_value(answer, v)]
        summary.update(conflicts=len(conflicts), conflicting_values=values, asserted=asserted)
        problems = []
        if not conflicts:
            problems.append("expected the agent to flag a conflict but conflicts is empty")
        if len(values) >= 2 and len(asserted) == 1:
            problems.append(f"answer asserts only one of the conflicting values ({asserted[0]!r})")
        if problems:
            return self.fail("; ".join(problems), sub_scores=[summary])
        note = "" if len(values) >= 2 else " (fewer than two conflicting values known; checked conflicts only)"
        return self.ok(f"flagged {len(conflicts)} conflict(s){note}", sub_scores=[summary])


class FollowsCorpus(ResearchCheck):
    """On `change_fact` cases, the answer follows the corpus, not the model's prior.

    Passes iff the answer matches the new value (`expected.answer` / aliases) and
    does not match `expected.original_answer`. On other perturbations it passes
    with reason "not applicable" and sub-scores `[{"applicable": false}]`; on
    change_fact the sub-score is `{"applicable": true, "matches_new", "matches_original", ...}`.
    """

    name = "follows_corpus"
    kind: CheckKind = "metamorphic"

    def applies(self, case: Case) -> bool:
        return _perturbation(case).get("op", "none") == "change_fact"

    def not_applicable_score(self, case: Case) -> Score:
        op = _perturbation(case).get("op", "none")
        return self.ok("not applicable", sub_scores=[{"applicable": False, "op": op}])

    def _score_research(self, case: Case, result: Result, output: dict[str, Any], ctx: CheckContext) -> Score:
        op = _perturbation(case).get("op", "none")
        if op != "change_fact":
            return self.ok("not applicable", sub_scores=[{"applicable": False, "op": op}])
        expected = _expected(case)
        original = expected.get("original_answer")
        if original in (None, ""):
            raise ValueError("change_fact case has no expected.original_answer")
        status = _status(output)
        answer = _answer(output) if status == "answered" else ""
        matches_new = status == "answered" and answer_matches(answer, expected)
        matches_original = status == "answered" and _matches_value(answer, str(original))
        detail = {
            "applicable": True,
            "status": status,
            "answer": truncate(answer, 300),
            "new": expected.get("answer"),
            "original": original,
            "matches_new": matches_new,
            "matches_original": matches_original,
        }
        if matches_new and not matches_original:
            return self.ok(f"answer follows the changed corpus ({expected.get('answer')!r})", sub_scores=[detail])
        if matches_original and not matches_new:
            reason = f"answer gives the original value {original!r}, not the corpus value {expected.get('answer')!r}"
        elif matches_original:
            reason = f"answer gives both the corpus value {expected.get('answer')!r} and the original {original!r}"
        elif status != "answered":
            reason = f"expected the corpus value {expected.get('answer')!r} but status is {status}"
        else:
            reason = f"answer {truncate(answer, 200)!r} matches neither the corpus value nor the original"
        return self.fail(reason, sub_scores=[detail])


class UncitedClaims(ResearchCheck):
    """At most `max_share` of claims have no citations.

    An answered output with zero claims counts as fully uncited (share 1.0); a
    not_found output with zero claims passes. `value` = share of cited claims.
    One sub-score per uncited claim: `{"claim": i, "text": str}`.
    """

    name = "uncited_claims"
    kind: CheckKind = "structural"

    def __init__(self, *, max_share: float, critical: bool) -> None:
        super().__init__(critical=critical)
        if isinstance(max_share, bool) or not isinstance(max_share, (int, float)) or not 0 <= max_share <= 1:
            raise TypeError(f"uncited_claims: max_share must be a number in [0, 1], got {max_share!r}")
        self.max_share = float(max_share)

    def _score_research(self, case: Case, result: Result, output: dict[str, Any], ctx: CheckContext) -> Score:
        status = _status(output)
        claims = _claims(output)
        if not claims:
            if status == "not_found":
                return self.ok("no claims (status not_found)")
            return self.fail("answered with no claims: the answer is entirely uncited", value=0.0)
        uncited = [{"claim": i, "text": truncate(c["text"], 300)} for i, c in enumerate(claims) if not c["citations"]]
        share = len(uncited) / len(claims)
        summary = f"{len(uncited)} of {len(claims)} claim(s) uncited ({share:.0%}, max {self.max_share:.0%})"
        if share <= self.max_share:
            return self.ok(summary, value=1 - share, sub_scores=uncited)
        return self.fail(summary, value=1 - share, sub_scores=uncited)


# --- claim_support (LLM judge) --------------------------------------------------


class ClaimVerdict(BaseModel):
    id: str
    supported: bool
    reason: str = ""


class ClaimSupportResponse(BaseModel):
    """The judge's response for task `claim_support` (docs/contracts.md)."""

    claims: list[ClaimVerdict] = Field(default_factory=list)


def _unit_hash(template_path: Path, system_template_path: Path | None) -> str:
    """`llm_evals.judges.client.unit_hash` (template bytes, then system bytes), or the
    same formula locally if the judges package doesn't import."""
    try:
        from llm_evals.judges.client import unit_hash
    except ImportError:
        h = hashlib.sha256(Path(template_path).read_bytes())
        if system_template_path is not None:
            h.update(Path(system_template_path).read_bytes())
        return h.hexdigest()[:12]
    return str(unit_hash(template_path, "", system_template=system_template_path))


def _is_judge_error(exc: BaseException) -> bool:
    """A `JudgeError` from the client (matched by class name, so fakes work too)."""
    return any(cls.__name__ == "JudgeError" for cls in type(exc).__mro__)


class _JudgeFailure(Exception):
    def __init__(self, reason: str, raw_text: str = "") -> None:
        super().__init__(reason)
        self.reason = reason
        self.raw_text = raw_text


class ClaimSupport(ResearchCheck):
    """Each claim is supported by the text of the documents it cites (LLM judge).

    "Supported" means the cited text states the claim or directly entails it.
    Claims with no citations, or citing only documents missing from the corpus,
    are unsupported without asking the judge. Sub-scores: one
    `{"unit": "claim_support", "passed": bool, "cost_usd": float}` followed by one
    `{"claim": i, "id": "c<i+1>", "supported": bool, "reason": str, "cited": [ids], "judged": bool}`
    per claim. Any failure without verdicts (no output, malformed output, judge
    error, infra, check error) returns a failed Score with a single
    `{"unit": "claim_support", "passed": False, "error": kind}` entry, like the
    other judge checks. Reasons: a `JudgeError`, an unusable response or a
    non-infra provider failure -> "judge error: ..."; an infra provider failure
    -> "infra: judge call failed (...)". A config problem (`FatalRunError`, e.g.
    a template needing an unknown variable) and an unbound judge raise.

    `prompt_hash` = `unit_hash(template, system_template=system_template)`: the
    user template's bytes then the system template's, so editing either prompt
    resets calibration (the judges module's convention).
    """

    name = "claim_support"
    kind: CheckKind = "faithfulness"
    needs_judge = True
    UNIT = "claim_support"

    def __init__(self, *, template: str, system_template: str, suite_dir: str | None, critical: bool) -> None:
        super().__init__(critical=critical)
        for label, value in (("template", template), ("system_template", system_template)):
            if not isinstance(value, str) or not value.strip():
                raise TypeError(f"claim_support: {label} must be a non-empty string, got {value!r}")
        if suite_dir is not None and not isinstance(suite_dir, (str, Path)):
            raise TypeError(f"claim_support: suite_dir must be a path string or None, got {suite_dir!r}")
        self.template = template
        self.system_template = system_template
        self.suite_dir = str(suite_dir) if suite_dir is not None else None
        self.judge: Any | None = None
        self._ctx_dir: Path | None = None

    def bind_judge(self, judge: Any) -> None:
        self.judge = judge

    def _base_dir(self) -> Path:
        if self.suite_dir is not None:
            return Path(self.suite_dir)
        if self._ctx_dir is not None:
            return self._ctx_dir
        return _DEFAULT_SUITE_DIR

    def template_path(self) -> Path:
        path = Path(self.template)
        return path if path.is_absolute() else self._base_dir() / path

    def system_template_path(self) -> Path:
        path = Path(self.system_template)
        return path if path.is_absolute() else self._base_dir() / path

    def prompt_hash(self) -> str:
        return _unit_hash(self.template_path(), self.system_template_path())

    def score(self, case: Case, result: Result, ctx: CheckContext) -> Score:
        if self.judge is None:
            raise RuntimeError("claim_support: no judge bound; call bind_judge(judge) first")
        ctx_dir = getattr(ctx, "suite_dir", None)
        if self.suite_dir is None and ctx_dir is not None:
            self._ctx_dir = Path(ctx_dir)
        score = super().score(case, result, ctx)
        if score.judge_model is None:
            score.judge_model = self._judge_model() or None
        if not any(isinstance(s, dict) and s.get("unit") == self.UNIT for s in score.sub_scores):
            reason = score.reason
            kind = next((k for p, k in (("judge error", "judge"), ("infra:", "infra"), ("no output", "no output"),
                                        ("malformed output", "malformed"), ("check error", "check"))
                         if reason.startswith(p)), "check")
            score.sub_scores = [{"unit": self.UNIT, "passed": False, "error": kind}, *score.sub_scores]
        return score

    def _judge_model(self) -> str:
        return str(getattr(self.judge, "model", "") or "")

    def gating_units(self) -> list[GatingUnit]:
        if self.judge is None:
            raise RuntimeError("claim_support: call bind_judge(judge) before gating_units()")
        return [GatingUnit(key=self.UNIT, judge_model=self._judge_model(), prompt_hash=self.prompt_hash())]

    def build_payload(self, case: Case, claims: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """The judge payload, and per-claim rows for claims settled without the judge."""
        corpus = corpus_docs(case) or {}
        judged: list[dict[str, Any]] = []
        local: list[dict[str, Any]] = []
        for i, claim in enumerate(claims):
            claim_id = f"c{i + 1}"
            docs = [
                {"id": doc_id, "text": str(corpus[doc_id].get("text", ""))}
                for doc_id in dict.fromkeys(claim["citations"])
                if doc_id in corpus
            ]
            if not docs:
                why = "no citations" if not claim["citations"] else "cited docs are not in the corpus"
                local.append({"claim": i, "id": claim_id, "supported": False, "reason": why,
                              "cited": claim["citations"], "judged": False})
                continue
            judged.append({"id": claim_id, "text": claim["text"], "cited_docs": docs})
        return {"task": "claim_support", "claims": judged}, local

    def _call(self, variables: dict[str, Any], payload: dict[str, Any]) -> tuple[ClaimSupportResponse, float]:
        """Run the judge; every non-config failure becomes a _JudgeFailure."""
        assert self.judge is not None
        try:
            judged = self.judge.judge(
                self.system_template_path(), self.template_path(), variables, payload, ClaimSupportResponse
            )
        except FatalRunError:
            raise
        except Exception as exc:
            if _is_judge_error(exc):
                raise _JudgeFailure(f"judge error: {exc}", str(getattr(exc, "raw_text", "") or "")) from exc
            kind = classify_exception(exc)  # raises FatalRunError for config problems
            if kind == ErrorKind.INFRA:
                raise _JudgeFailure(f"infra: judge call failed ({type(exc).__name__}: {exc})") from exc
            raise _JudgeFailure(f"judge error: judge call failed ({type(exc).__name__}: {exc})") from exc
        raw = str(getattr(judged, "raw_text", None) or getattr(judged, "raw", None) or "")
        try:
            parsed = judged.parsed
            if isinstance(parsed, BaseModel) and not isinstance(parsed, ClaimSupportResponse):
                parsed = parsed.model_dump()
            if not isinstance(parsed, ClaimSupportResponse):
                parsed = ClaimSupportResponse.model_validate(parsed)
            response = parsed
        except Exception as exc:
            raise _JudgeFailure(f"judge error: unusable response ({type(exc).__name__}: {exc})", raw) from exc
        return response, float(getattr(judged, "cost_usd", 0.0) or 0.0)

    def _score_research(self, case: Case, result: Result, output: dict[str, Any], ctx: CheckContext) -> Score:
        judge_model = self._judge_model() or None
        prompt_hash = self.prompt_hash()
        claims = _claims(output)
        status = _status(output)
        fields: dict[str, Any] = {"judge_model": judge_model, "prompt_hash": prompt_hash}

        if not claims:
            return self.ok(
                f"no claims to check (status {status})",
                sub_scores=[{"unit": self.UNIT, "passed": True, "cost_usd": 0.0}],
                **fields,
            )

        payload, local_rows = self.build_payload(case, claims)
        rows: dict[int, dict[str, Any]] = {row["claim"]: row for row in local_rows}
        cost = 0.0
        if payload["claims"]:
            variables = {
                "question": str((case.input or {}).get("question", "")),
                "answer": _answer(output),
                "n_claims": len(payload["claims"]),
            }
            try:
                response, cost = self._call(variables, payload)
            except _JudgeFailure as exc:
                subs: list[dict[str, Any]] = []
                if exc.raw_text:
                    subs.append({"raw_judge_output": exc.raw_text[:2000]})
                return self.fail(exc.reason, sub_scores=subs, **fields)
            verdicts = {v.id: v for v in response.claims}
            missing = [c["id"] for c in payload["claims"] if c["id"] not in verdicts]
            if missing:
                return self.fail(f"judge error: no verdict for claim(s) {', '.join(missing)}", **fields)
            for i, claim in enumerate(claims):
                if i in rows:
                    continue
                verdict = verdicts[f"c{i + 1}"]
                rows[i] = {"claim": i, "id": verdict.id, "supported": verdict.supported,
                           "reason": verdict.reason, "cited": claim["citations"], "judged": True}

        ordered = [rows[i] | {"text": truncate(claims[i]["text"], 300)} for i in sorted(rows)]
        unsupported = [r for r in ordered if not r["supported"]]
        passed = not unsupported
        sub_scores = [{"unit": self.UNIT, "passed": passed, "cost_usd": cost}, *ordered]
        value = 1 - len(unsupported) / len(ordered)
        if passed:
            return self.ok(f"all {len(ordered)} claim(s) supported by their cited docs",
                           value=value, sub_scores=sub_scores, **fields)
        detail = "; ".join(f"{r['id']}: {r['reason'] or 'unsupported'}" for r in unsupported)
        return self.fail(f"{len(unsupported)} of {len(ordered)} claim(s) unsupported ({truncate(detail, 600)})",
                         value=value, sub_scores=sub_scores, **fields)


# --- Factories ----------------------------------------------------------------


@register_check("citations_exist")
def citations_exist(*, critical: bool = True) -> CitationsExist:
    return CitationsExist(critical=critical)


@register_check("answer_match")
def answer_match(*, critical: bool = False) -> AnswerMatch:
    return AnswerMatch(critical=critical)


class BehaviorMatch(AnswerMatch):
    """`answer_match` restricted to one expected behavior.

    One check per case type keeps a regression on one type (e.g. no longer
    abstaining when the answer was removed) from being diluted, or masked by
    gains on another type, inside a single answer_match pass rate. Cases of
    other types pass as not applicable, including when they errored, so an
    error is counted once, by the check for its own case type.
    """

    def __init__(self, *, name: str, behavior: str, critical: bool) -> None:
        super().__init__(critical=critical)
        self.name = name
        self.behavior = behavior

    def applies(self, case: Case) -> bool:
        return _expected(case).get("behavior", "answer") == self.behavior


@register_check("answer_correct")
def answer_correct(*, critical: bool = False) -> BehaviorMatch:
    """Answerable questions: the answer matches."""
    return BehaviorMatch(name="answer_correct", behavior="answer", critical=critical)


@register_check("abstains_when_missing")
def abstains_when_missing(*, critical: bool = False) -> BehaviorMatch:
    """The answer was removed from the corpus: status must be not_found."""
    return BehaviorMatch(name="abstains_when_missing", behavior="abstain", critical=critical)


@register_check("flags_conflicts")
def flags_conflicts(*, critical: bool = False) -> BehaviorMatch:
    """Sources disagree: the conflict is reported, not resolved by picking one."""
    return BehaviorMatch(name="flags_conflicts", behavior="flag_conflict", critical=critical)


@register_check("follows_corpus")
def follows_corpus(*, critical: bool = False) -> FollowsCorpus:
    return FollowsCorpus(critical=critical)


@register_check("uncited_claims")
def uncited_claims(*, max_share: float = 0.1, critical: bool = False) -> UncitedClaims:
    return UncitedClaims(max_share=max_share, critical=critical)


@register_check("claim_support")
def claim_support(
    *,
    template: str = "judges/claim_support.jinja",
    system_template: str = "judges/claim_support_system.jinja",
    suite_dir: str | None = None,
    critical: bool = False,
) -> ClaimSupport:
    return ClaimSupport(template=template, system_template=system_template, suite_dir=suite_dir, critical=critical)
