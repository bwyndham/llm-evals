# llm-evals

Paired regression evals for LLM features. Every change (prompt, skill, model or code)
runs against the current baseline on the same cases, with fresh samples for both.
The gate decides from the 95% confidence interval of the change in case pass rate:

| Decision | When | Exit code |
| --- | --- | --- |
| PROMOTE | CI lower bound above −tolerance, and no critical check newly failing | 0 |
| HOLD | Inconclusive | 1 |
| ROLLBACK | CI upper bound below 0, or a critical check newly failing | 2 |
| INCOMPLETE | Too many infra errors, or the cost cap stopped the run | 3 |

## Status

Version 0.1, [MIT licensed](LICENSE). Maintained as a portfolio project: issues and
questions are welcome, with no promise of support. What it doesn't do yet: models other than Claude; multi-turn conversations (it runs
one-shot prompts and tool-using agents); and the faithfulness judge in the example
suite hasn't passed its own calibration bar, so it reports but doesn't gate.

## Use it in your own repo

Install the `llm-evals` command:

```bash
uv tool install git+https://github.com/bwyndham/llm-evals@v0.1.0
# or from a local clone: uv tool install ./llm-evals
```

Then:

```bash
llm-evals init evals/my-feature --ci       # a starter suite and a PR workflow
llm-evals gate evals/my-feature --candidate versions/v2.yaml --provider mock --skip-judges
```

That last line runs offline against a stand-in model: no API key, no cost. Bring your
own prompts, cases with known answers, checks and LLM judges (as plug-ins listed in
`suite.yaml`), and gate pull requests with this repo's GitHub Action.
[docs/adopting.md](docs/adopting.md) walks through it.

## Quick start (this repo)

```bash
uv sync
uv run pytest                                   # no API key needed
uv run eval estimate suites/spec-writing        # calls, tokens and dollars before anything runs
uv run eval compare suites/spec-writing --candidate versions/seeded_no_split_bundles.yaml --limit 20 --out out/
uv run eval seeded suites/spec-writing --limit 30
uv run eval aa suites/spec-writing --runs 20 --limit 30
uv run eval promote suites/spec-writing versions/baseline.yaml
```

Model calls go through the Anthropic SDK. Put your key in a gitignored `.env`
at the repo root (copy `.env.example`); a key exported in your shell wins.

### Offline demo mode (no API key)

`--provider mock` swaps in rule-based stand-ins for the spec writer, the research
agent and the judges. Their output depends on the prompt, so degraded prompts
really are caught. This shows the pipeline working, not model quality.

```bash
uv run eval seeded suites/spec-writing --provider mock --limit 40
uv run eval seeded suites/research-agent --provider mock
uv run eval compare suites/research-agent --provider mock --candidate versions/seeded_ignore_corpus.yaml
```

### Judges and calibration

LLM judge checks are scored and reported on every run, but a judge unit gates only
after a human-labeled calibration is approved (n ≥ 50, Cohen's κ ≥ 0.6) for that
exact judge model and prompt. `--skip-judges` leaves them out for cheap runs.

```bash
uv run eval calibration label --db suites/spec-writing/.evals/evals.sqlite --run <run id> --check faithfulness --labeler you
uv run eval calibration calibrate --db suites/spec-writing/.evals/evals.sqlite --check faithfulness
```

## Layout

- `src/llm_evals/types.py`: shared types. `registry.py`: plug-in registries.
- `engine.py`: the comparison. `qa.py`: estimate, A/A and seeded-regression tools.
- `stats/`, `runners/`, `checks/`, `cases/`, `store/`, `report/`, `cli/`.
- `suites/spec-writing/`: pilot 1 (requirement atoms → messy asks → tickets).
- `suites/research-agent/`: pilot 2 (questions over frozen synthetic corpora with planted, removed, changed and conflicting facts).
- `docs/contracts.md`: module contracts. The PRD is a Claude Doc linked in `CLAUDE.md`.
