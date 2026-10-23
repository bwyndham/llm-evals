"""Plumbing shared by the judge-based checks (faithfulness, checklist, pairwise).

Template and rubric paths are relative to the suite dir. How the suite dir
reaches a check:

- Preferred: the engine passes `suite_dir` as a constructor kwarg. Paths are
  resolved and checked for existence at construction, and `gating_units()`
  works as soon as a judge is bound.
- Fallback: with no `suite_dir`, `score()` resolves paths against
  `ctx.suite_dir` on first use and remembers it, and `gating_units()` raises
  until then. (Absolute paths are used as given, but a suite dir is still
  needed to trigger resolution.)

Subclasses set their own path fields before calling `_init_judge`, because
it may call `_resolve`, which they extend.

This module deliberately doesn't import `llm_evals.checks`, so the checks
package can import it while it is itself still initialising.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel

from llm_evals.judges.client import JudgeClient, JudgeError, JudgeResult, unit_hash
from llm_evals.runners.errors import FatalRunError, classify_exception
from llm_evals.types import Case, CheckContext, ErrorKind, Result, Score


class JudgeFailure(Exception):
    """A judge call that produced no usable verdict. `reason` is Score-ready:
    it starts with "judge error:" (counts as a failure) or "infra:" (the engine
    may exclude the sample)."""

    def __init__(self, reason: str, *, raw_text: str = "") -> None:
        super().__init__(reason)
        self.reason = reason
        self.raw_text = raw_text


def load_rubric(path: Path) -> list[dict[str, str]]:
    """Read a rubric YAML: a list of `{id, question}` (or `{items: [...]}`)."""
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FatalRunError(f"rubric not found: {path}") from None
    if isinstance(data, dict):
        data = data.get("items")
    if not isinstance(data, list) or not data:
        raise FatalRunError(f"rubric {path} must be a non-empty list of {{id, question}} items")
    items: list[dict[str, str]] = []
    seen: set[str] = set()
    for i, entry in enumerate(data):
        if not isinstance(entry, dict) or not isinstance(entry.get("id"), str) or not isinstance(
            entry.get("question"), str
        ):
            raise FatalRunError(f"rubric {path} item {i} needs string `id` and `question`, got {entry!r}")
        item_id, question = entry["id"].strip(), entry["question"].strip()
        if not item_id or not question:
            raise FatalRunError(f"rubric {path} item {i} has an empty id or question")
        if item_id in seen:
            raise FatalRunError(f"rubric {path} repeats item id {item_id!r}")
        seen.add(item_id)
        items.append({"id": item_id, "question": question})
    return items


class JudgeBinding:
    """Judge binding, suite-relative paths, and judge-call error handling."""

    name: str
    needs_judge = True

    def _init_judge(self, *, template: str, system_template: str | None, suite_dir: str | Path | None) -> None:
        self.template = template
        self.system_template = system_template
        self.suite_dir: Path | None = None
        self.judge: JudgeClient | None = None
        self.template_path: Path | None = None
        self.system_template_path: Path | None = None
        if suite_dir is not None:
            self._resolve(Path(suite_dir))

    # --- paths ------------------------------------------------------------

    def _path(self, base: Path, rel: str) -> Path:
        p = Path(rel)
        p = p if p.is_absolute() else base / p
        if not p.is_file():
            raise FatalRunError(f"{self.name}: file not found: {p} (paths are relative to the suite dir {base})")
        return p

    def _resolve(self, suite_dir: Path) -> None:
        """Resolve every path against `suite_dir`; subclasses extend this."""
        self.suite_dir = suite_dir
        self.template_path = self._path(suite_dir, self.template)
        self.system_template_path = (
            self._path(suite_dir, self.system_template) if self.system_template is not None else None
        )

    def _ensure_resolved(self, suite_dir: Path | None = None) -> None:
        if self.template_path is not None:
            return
        if suite_dir is None:
            raise RuntimeError(
                f"{self.name}: no suite dir yet; pass suite_dir= when constructing the check "
                "(or score once with a context) before asking for gating units"
            )
        self._resolve(Path(suite_dir))

    # --- judge ------------------------------------------------------------

    def bind_judge(self, judge: JudgeClient) -> None:
        self.judge = judge

    def _require_judge(self) -> JudgeClient:
        if self.judge is None:
            raise RuntimeError(f"{self.name}: no judge bound; call bind_judge(JudgeClient(...)) before scoring")
        return self.judge

    @property
    def judge_model(self) -> str:
        return self._require_judge().model

    def unit_hash(self, item_text: str = "") -> str:
        self._ensure_resolved()
        assert self.template_path is not None
        return unit_hash(self.template_path, item_text, system_template=self.system_template_path)

    def _call(self, payload: dict[str, Any], response_model: type[BaseModel]) -> JudgeResult[Any]:
        """Run the judge; turn every non-config failure into a JudgeFailure."""
        judge = self._require_judge()
        assert self.template_path is not None
        try:
            return judge.judge(self.system_template_path, self.template_path, {}, payload, response_model)
        except JudgeError as exc:
            raise JudgeFailure(f"judge error: {exc}", raw_text=exc.raw_text) from exc
        except FatalRunError:
            raise
        except Exception as exc:  # noqa: BLE001 - classify provider failures
            kind = classify_exception(exc)  # raises FatalRunError for config problems
            if kind == ErrorKind.INFRA:
                raise JudgeFailure(f"infra: judge call failed ({type(exc).__name__}: {exc})") from exc
            raise JudgeFailure(f"judge error: judge call failed ({type(exc).__name__}: {exc})") from exc


class JudgeCheckMixin(JudgeBinding):
    """A judge-based Check: combine with BaseCheck (`class X(JudgeCheckMixin, BaseCheck)`).

    Overrides `score` and implements `unit_keys` / `failed_units`; subclasses
    implement `_score` and `gating_units`.
    """

    def unit_keys(self) -> list[str]:
        raise NotImplementedError

    def failed_units(self, error: str) -> list[dict[str, Any]]:
        """One failed unit entry per unit, for results with no verdict."""
        return [{"unit": key, "passed": False, "error": error} for key in self.unit_keys()]

    def score(self, case: Case, result: Result, ctx: CheckContext) -> Score:
        """Like BaseCheck.score, but an unbound judge or a config problem
        (FatalRunError) raises instead of becoming a failed Score."""
        from llm_evals.checks._base import no_output_reason  # lazy: see module docstring

        self._require_judge()
        self._ensure_resolved(getattr(ctx, "suite_dir", None))
        fail = self.fail  # type: ignore[attr-defined]  # from BaseCheck
        missing = no_output_reason(result)
        if missing is not None:
            return fail(missing, sub_scores=self.failed_units("no output"), judge_model=self.judge_model)
        try:
            return self._score(case, result.output or {}, ctx)  # type: ignore[attr-defined]
        except FatalRunError:
            raise
        except JudgeFailure as exc:
            kind = "infra" if exc.reason.startswith("infra:") else "judge"
            subs = self.failed_units(kind)
            if exc.raw_text:
                subs.append({"raw_judge_output": exc.raw_text[:2000]})
            return fail(exc.reason, sub_scores=subs, judge_model=self.judge_model)
        except Exception as exc:  # noqa: BLE001 - a check must never raise
            return fail(
                f"check error: {type(exc).__name__}: {exc}",
                sub_scores=self.failed_units("check"),
                judge_model=self.judge_model,
            )


__all__ = ["JudgeBinding", "JudgeCheckMixin", "JudgeFailure", "load_rubric"]
