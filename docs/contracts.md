# Module contracts (Phase 1)

Requirement ids like `CAL-4` or `STAT-9` refer to the original design document, which
isn't part of this repository. This file holds the parts every module needs to agree on.
Shared types live in `src/llm_evals/types.py`, and plug-in registration lives in
`src/llm_evals/registry.py`.

## Layout and ownership

| Path | Owner | Contents |
|---|---|---|
| `src/llm_evals/types.py`, `registry.py` | core | Shared types and plug-in registries. Don't edit them from a module; propose changes instead. |
| `src/llm_evals/stats/` | stats | Paired Δ, cluster bootstrap, Wilson, McNemar, MDE, gate decision |
| `src/llm_evals/runners/simulated.py` | stats | QA-1 simulated runner |
| `src/llm_evals/cases/` | cases | Case loading, validation, case-set hash, renderers |
| `suites/spec-writing/` | cases | Atoms data, spec-writer prompt, seeded variants, suite.yaml |
| `src/llm_evals/runners/` (except simulated.py), `store/` | runners | Version manifest, prompt runner, Anthropic adapter, error classification, cache, SQLite store |
| `src/llm_evals/checks/` | checks | Deterministic and metamorphic checks for spec writing |
| `src/llm_evals/cli/`, `report/`, engine | core | Orchestration, reports, CLI, written after the modules |

Each package's `__init__.py` must import its plug-in modules so that their
`@register_*` decorators run. `registry.load_builtins()` imports
`llm_evals.runners`, `llm_evals.checks` and `llm_evals.cases.renderers`.

Rules for all modules:
- Python 3.11, type hints throughout, pydantic models from `types.py`.
- Don't edit `pyproject.toml`. If you need a dependency, say so in your report.
  Already installed: anthropic, pydantic, pyyaml, jinja2, numpy, scipy, typer,
  gherkin-official, pytest.
- Tests go in `tests/<area>/test_*.py` and run with `uv run pytest`.
- Tests never call a real model API. No API key is set.
- Every random step takes an explicit seed.
- Don't commit. The core owner reviews and commits.

## Spec writer (the system under test for the spec-writing pilot)

Input, in `case.input`: `{"ask": "<messy request text>"}`.

Output: a JSON object the prompt runner parses into `result.output`:

```json
{
  "title": "string, 80 chars or fewer",
  "description": "string",
  "acceptance_criteria": "Gherkin text: one Feature with one or more Scenarios using Given/When/Then",
  "assumptions": ["string"],
  "open_questions": ["string"]
}
```

The prompt asks for exactly this JSON and nothing else. If the model returns
anything that isn't valid JSON matching this shape, it's a SYSTEM error.

## Requirement atoms (constructed cases)

`suites/spec-writing/data/atoms.jsonl` holds one atom set per line:

```json
{"id": "export-csv", "domain": "reporting", "actor": "account admin",
 "atoms": [{"id": "a1", "text": "Admins can export the usage report as CSV", "kind": "functional"},
           {"id": "a2", "text": "The export covers the last 90 days", "kind": "constraint"}]}
```

The `ask_renderer` turns an atom set plus features into a Case:
- `input = {"ask": text}`
- `source = {"atoms": [...], "actor": ...}`
- `strata` = the feature values
- `provenance = "constructed"`
- `id` = a stable hash of (atom set id, features, seed)

## Stats API (consumed by the engine)

```python
# llm_evals.stats
Outcomes = dict[str, list[bool]]  # case_id -> pass/fail per sample, infra-error samples removed

def compare(baseline: Outcomes, candidate: Outcomes, *, n_boot: int = 10_000,
            seed: int = 0, alpha: float = 0.05, power: float = 0.80) -> PairedComparison
    # Uses only case ids present in both. PairedComparison fields: delta, ci_low, ci_high,
    # n_cases, k_baseline, k_candidate, baseline_rate, candidate_rate, mde, mcnemar_p,
    # flips: list[Flip(case_id, direction: "broke" | "fixed", baseline_rate, candidate_rate)]

def decide(cmp: PairedComparison, *, tolerance: float, critical_new_failures: list[str],
           incomplete: bool) -> Decision  # from llm_evals.types
```

Gate rule (PRD, Statistics and gate):
- INCOMPLETE if `incomplete` is true.
- ROLLBACK if `ci_high < 0`, or `critical_new_failures` is non-empty.
- PROMOTE if `ci_low > -tolerance`.
- Otherwise HOLD.

The Δ CI comes from a cluster bootstrap that resamples cases with replacement
and keeps all k samples of a case together. A case passes when every gating
check passes. A flip compares majority outcomes: "broke" means the case passed
on most baseline samples and failed on most candidate samples.

# Phase 2 and 3 contracts

## New ownership

| Path | Owner | Contents |
|---|---|---|
| `src/llm_evals/judges/`, `checks/faithfulness.py`, `checks/checklist.py`, `checks/pairwise.py`, `suites/spec-writing/judges/`, `suites/spec-writing/rubrics/` | judges | JudgeClient with cache, LLM judge checks, judge prompts, rubric |
| `src/llm_evals/calibration/`, `src/llm_evals/cli/calibrate.py`, new tables and methods in `store/sqlite.py` | calibration | Labels, Cohen's κ, sensitivity/specificity, approved calibrations |
| `src/llm_evals/runners/providers/mock.py`, `checks/atom_keywords.py` | offline | Rule-based spec writer and judge for offline demo mode; deterministic keyword faithfulness check |
| `suites/research-agent/`, `src/llm_evals/cases/corpus.py` | research-data | Synthetic corpora, world facts, questions, corpus perturbation (CASE-6), prompts, manifests, suite.yaml |
| `src/llm_evals/agents/`, `runners/agent_replay.py`, `runners/providers/tools.py`, `runners/providers/anthropic_tools.py`, `runners/providers/mock_research.py` | research-agent | The research agent, tool-use loop, replay runner (RUN-7), tool providers (real and mock) |
| `checks/research.py` | research-checks | citations_exist, answer_match, follows_corpus, uncited_claims, claim_support (CHK-9) |
| engine, `cli/main.py`, `report/` | core | Wiring, done after the modules |

The `checks/__init__.py` and `runners/__init__.py` files are shared: add ONLY a
single `from llm_evals.checks import <your_module>  # noqa: F401` (or runners)
line for each module you add, at the end of the import block. Don't reorder or
remove other lines.

## Judge checks and gating

- A judge-based check has `needs_judge = True` and `bind_judge(judge: JudgeClient) -> None`.
  The engine constructs checks from YAML params, then calls `bind_judge`.
- It exposes `gating_units() -> list[GatingUnit]` (`GatingUnit` is in types.py).
  The unit `prompt_hash` = sha256(template file bytes + item text)[:12], fixed for a
  given template and rubric item, never per case.
- Its `Score.sub_scores` include one `{"unit": key, "passed": bool, ...}` per unit,
  and `Score.passed` = every unit passed.
- The engine counts a unit toward the case pass only if
  `store.approved_calibration(key, judge_model, prompt_hash)` is true (default bar
  n ≥ 50 labels and κ ≥ 0.6). Other units are reported but don't gate.

`JudgeClient(provider: Provider, *, model="claude-sonnet-5-5", params=None, cache_path=None)`
renders a judge template (system + user), calls the text `Provider`, parses JSON
into a pydantic model, and caches by prompt hash (model + rendered prompt) in its
own SQLite file. Judge calls report cost so the engine can total them.

## Judge payload convention (so the offline mock judge can answer)

Every judge user prompt ends with one fenced block holding a JSON payload:

    ```json
    {"task": "...", ...}
    ```

The mock judge reads the LAST fenced json block. Payloads and responses:

| task | payload | response |
|---|---|---|
| `faithfulness` | `{"atoms": [{"id", "text"}], "ticket": {...}}` | `{"atoms": [{"id", "covered": bool, "evidence": str}], "invented": [{"text", "reason"}]}` |
| `checklist` | `{"items": [{"id", "question"}], "ask": str, "ticket": {...}}` | `{"items": [{"id", "answer": "yes"|"no", "reason": str}]}` |
| `pairwise` | `{"criteria": [str], "ask": str, "a": {...}, "b": {...}}` | `{"winner": "a"|"b"|"tie", "reason": str}` |
| `claim_support` | `{"claims": [{"id", "text", "cited_docs": [{"id", "text"}]}]}` | `{"claims": [{"id", "supported": bool, "reason": str}]}` |

## Keyword faithfulness (deterministic, `atom_keywords`)

Ticket text = title + description + acceptance_criteria (assumptions and open
questions excluded). An atom is covered if at least ceil(0.6 × len(keywords)) of
its `keywords` appear (case-insensitive substring) in the ticket text. A Scenario
whose name and step text contain none of the keywords of any atom in the case is
an invented requirement. The check fails when any non-hinted atom is uncovered,
or any scenario is invented.

## Research agent

Case: `input = {"question": str}`; `fixtures = {"corpus_id": str, "docs": [{"id", "title", "text"}]}`;
`expected = {"behavior": "answer"|"abstain"|"flag_conflict", "answer": str|None, "aliases": [str], "original_answer": str|None}`;
`source = {"facts": [...], "perturbation": {"op": "none"|"plant_fact"|"remove_answer_doc"|"change_fact"|"add_conflict", ...}}`.

Tools the agent can call (served by the replay layer from `case.fixtures`):
- `search(query: str) -> [{"doc_id", "title", "snippet"}]` — deterministic keyword
  ranking over the case's docs, top 5.
- `fetch(doc_id: str) -> {"doc_id", "title", "text"}` — an unknown id is an
  "unrecorded call": fail the sample (SYSTEM) or return an error result, per
  `on_unrecorded_call`.

Agent output (`result.output`):

    {"status": "answered"|"not_found", "answer": str,
     "claims": [{"text": str, "citations": [doc_id]}], "conflicts": [str]}

`result.trace.tool_calls` = `[{"name", "input", "doc_ids": [...returned or fetched]}]`.

Runner: `@register_runner("agent_replay") AgentReplayRunner(tool_provider, entry: str, on_unrecorded_call="fail", max_turns=8)`;
`entry` is `"module:function"` with signature
`run(case_input: dict, tools: ToolSession, provider: ToolProvider, version: VersionManifest) -> AgentOutcome`.

`ToolProvider.create(*, model, system, messages, tools, params) -> ToolResponse`
(content blocks: text / tool_use, stop_reason, input/output tokens, request_id).
`AnthropicToolProvider` wraps `client.messages.create(..., tools=...)`.
`MockResearchModel` is rule-based for offline mode.
