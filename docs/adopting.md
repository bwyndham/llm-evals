# Adding llm-evals to your repo

llm-evals gates changes to an LLM feature the way tests gate code. Every pull
request that changes a prompt, model or parameters runs the base branch's
version and the PR's version on the same cases, several times each, and gets
**PROMOTE**, **HOLD** or **ROLLBACK** from a paired statistical comparison, with
the evidence in a PR comment.

It fits when you can say, for a set of inputs, what a good output must do: pick
the right category, cite a source, stay under a length, never promise a refund.
It doesn't replace judgment calls with no checkable answer, though an LLM
judge you've calibrated against your own labels can cover some of them.

> **Status.** Version 0.1, MIT licensed. Pin to the `v0.1.0` tag, as below:
> `main` may change.

## 1. Install

```bash
uv tool install git+https://github.com/bwyndham/llm-evals@v0.1.0
# or: pipx install git+https://github.com/bwyndham/llm-evals@v0.1.0
# or, from a local clone: uv tool install ./llm-evals
```

The command is `llm-evals`. (Inside this repo it also answers to `eval`, run
through `uv run`; in a plain shell, `eval` is bash's own built-in.)

It needs Python 3.11+ and, for real runs, `ANTHROPIC_API_KEY`. Claude models only.

## 2. Start from a working suite

```bash
llm-evals init evals/support-triage --ci
llm-evals gate evals/support-triage --candidate versions/v2.yaml --provider mock --skip-judges
```

`init` writes a small, complete suite: it triages support messages into a
category and drafts a reply. `--provider mock` answers with the stand-in in
`plugins/mock.py`, so the whole pipeline runs offline and free. That proves the
wiring, not quality. `--ci` also writes `.github/workflows/llm-evals.yml`.

## 3. Anatomy of a suite

A suite is a folder with a `suite.yaml`:

| Key | What it is |
|---|---|
| `suite` | A name for reports and PR comments. |
| `runner` | How the system under test runs. `{type: prompt, output_schema: ...}` makes one prompt call per case. |
| `baseline` | The version file CI compares a PR against. |
| `cases` | `- from: data/cases.jsonl`, one or more JSONL files. |
| `checks` | The checks, by name, with their settings. |
| `plugins` | Your own checks, judges, runners and schemas: `.py` files next to the suite, or importable modules. |
| `mock` | `file.py:function` that answers instead of the model under `--provider mock`. |
| `samples` | How many times each case runs per version. |
| `gate` | `tolerance` (how much worse counts as a regression) and other gate settings. |
| `judge` | The judge model, for LLM-judge checks. |

### Versions

A version is everything the model sees: model, parameters and prompt templates
(Jinja, with the case's `input` as variables). Its id is a hash of all of it, so
editing a template's bytes is a new version, and a PR that changes nothing the
model sees is skipped in CI.

```yaml
name: v2
model: claude-haiku-5-5
prompt_templates:
  system: prompts/system_v2.jinja
  user: prompts/user.jinja
params:
  max_tokens: 1000
```

### Cases

One JSON object per line:

```json
{"id": "bill-double", "input": {"message": "I was charged twice..."}, "expected": {"category": "billing", "keywords": ["refund"]}, "strata": {"channel": "email"}, "provenance": "constructed"}
```

- `input`: the template variables.
- `expected`: the known answer that checks compare against.
- `strata`: labels to slice results by.
- `provenance`: where the case came from (`constructed`, `production`, `public`, ...).

Cases with known answers are what make the gate trustworthy. A few dozen
well-chosen cases beat hundreds of unlabeled ones.

### Output schemas

Point `runner.output_schema` at a JSON Schema file in the suite (`.json`,
`.yaml` or `.yml`). Output that isn't valid JSON, or doesn't match, counts as a
failed sample. If you prefer, register a pydantic model with
`@register_schema("name")` in a plug-in and use its name.

## 4. Checks

### Built in

| Check | Passes when | Settings |
|---|---|---|
| `field_matches` | an output field equals the case's expected value (or one of a list) | `field`, `expected_key`, `ignore_case` |
| `keywords_present` | a field contains the case's expected keywords | `field`, `expected_key` (default `keywords`), `min_share` |
| `regex` | a field matches a pattern, or doesn't | `field`, `pattern`, `must_match`, `flags` (`ims`) |
| `max_length` | a field is at most N characters | `field`, `max_chars` |
| `checklist` | an LLM judge answers "yes" to each rubric question | `rubric`, `template`, `system_template` |
| `pairwise` | (report only) a judge prefers one version's output to the other's | `rubric` |

`field` takes dots for nested output (`answer.label`). Cases without the
expected key a check needs pass as "not applicable" and don't count.

Two settings work on any check:
- `critical: true` makes a case that newly fails on every sample, confirmed by reruns, roll back the PR on its own.
- `gate: false` keeps a check in the report but out of the decision.

### Your own

A check is a class with `_score(case, output, ctx)` that returns a `Score`, and
a factory registered by name:

```python
# plugins/checks.py
from llm_evals.api import BaseCheck, Score, register_check

class CitesOrder(BaseCheck):
    name = "cites_order"
    kind = "deterministic"

    def _score(self, case, output, ctx):
        order = case.input.get("order_id")
        if order and order not in output.get("reply", ""):
            return self.fail(f"reply doesn't mention order {order}")
        return Score(check=self.name, kind=self.kind, passed=True, reason="ok")

@register_check("cites_order")
def cites_order(*, critical: bool = False):
    return CitesOrder(critical=critical)
```

List the file under `plugins:` and the check under `checks:`. `BaseCheck`
already turns failed calls and exceptions into failed scores. Override
`applies(case)` to skip cases a check doesn't measure.

## 5. LLM judges

Some things have no keyword: is the reply helpful, did the summary invent a
fact? An LLM judge can grade those. But a judge is a measuring instrument, and
an uncalibrated one moves your numbers in ways you can't see. So in llm-evals,
**a judge is scored on every run but gates only after it agrees with you**.

1. **Use the built-in `checklist`** with your own yes/no rubric
   (`rubrics/*.yaml`, each question phrased so "yes" is good), or
   **write your own** judge. `docs/examples/polite_judge.py` is a complete one:
   one question, one gating unit.
2. **Run it** (`llm-evals gate ...` without `--skip-judges`), then **label** a
   sample of its outputs blind to its verdicts:
   `llm-evals calibration label --db <suite>/.evals/evals.sqlite --run <run id> --check checklist --labeler you`
   (or `calibration serve` for a local web page).
3. **Calibrate**: `llm-evals calibration calibrate --db ... --check checklist`.
   A unit gates once it has at least 50 labels and Cohen's κ of at least 0.6,
   for that exact judge model and prompt. Change the prompt and it must earn
   its place again.

Back up your labels with `calibration backup-labels`. They're the expensive part.

## 6. Set the tolerance from measured noise

The model answers differently every time, so a no-op change still moves the
numbers. Measure how much before you trust the gate:

```bash
llm-evals estimate evals/support-triage          # cost and time, no calls
llm-evals aa evals/support-triage --runs 10      # the same version against itself
```

`aa` reports the **smallest drop the suite can reliably detect** when nothing
changed: the noise floor. If that's above your tolerance, the gate will often say
HOLD on harmless changes. More `samples` per case or more cases lower it: noise
falls with the square root of cases × samples. In this repo, going from 3 to 6
samples on 120 cases took the detectable drop from 6.8 to 5.1 points. In the demo
suite, 10 A/A runs at 3 samples put it at about 2 points, so a 3% tolerance fit.

A HOLD on a pull request also gives the detectable drop for that run. Read it as
a hint, not the noise floor: it comes from that run's own spread, which a large or
uneven change inflates.

Prove it catches what you care about, too. Write a deliberately worse prompt
version, list it under `seeded_variants`, and check that `llm-evals seeded`
rolls it back.

## 7. Gate pull requests

```yaml
# .github/workflows/llm-evals.yml (written by `llm-evals init --ci`)
on:
  pull_request:
    paths: ["evals/support-triage/**"]
permissions:
  contents: read
  pull-requests: write
jobs:
  gate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: bwyndham/llm-evals@v0.1.0
        with:
          suite: evals/support-triage
          anthropic-api-key: ${{ secrets.ANTHROPIC_API_KEY }}
```

The action runs the PR's version against the base branch's and posts one
comment per suite, edited on each push. The check **passes on PROMOTE or HOLD**
(the comment says which) and **fails on ROLLBACK, INCOMPLETE or a crash**. It
also skips the paid run when the version didn't change.

| Input | Default | |
|---|---|---|
| `suite` | required | Suite folder, relative to the repo root |
| `anthropic-api-key` | required | A repository secret |
| `cost-cap` | `5` | Stop once spend passes this many dollars |
| `concurrency` | `8` | Model calls in flight |
| `skip-judges` | `true` | Leave out LLM judges |
| `skip-unchanged` | `true` | Skip when the version didn't change |
| `report-name` | `llm-evals-report` | Artifact name; set one per suite |

The `decision` output is `PROMOTE`, `HOLD`, `ROLLBACK`, `INCOMPLETE`, `SKIPPED` or `ERROR`.

**To vendor the Action** (keep a copy inside your own repo instead of referencing
this one), copy the whole llm-evals folder, not just `action.yml`: the Action
installs llm-evals from that copy.

```bash
mkdir -p .github/actions && cp -R path/to/llm-evals .github/actions/llm-evals
llm-evals init evals/support-triage --ci --action ./.github/actions/llm-evals
```

The workflow then says `uses: ./.github/actions/llm-evals`, and everything else is the same.

## Limits

- **Claude only.** A provider interface exists (`llm_evals.runners.providers.base`),
  but only Anthropic and the offline mock implement it.
- **You pay for runs.** About `cases × samples × 2` model calls per gated PR.
  Run `estimate` first, and keep a `cost-cap`.
- **Known answers are work.** The gate is only as good as the cases and checks
  behind it.
- **Judges need your labels** before they can gate: about 50 per judge question.
