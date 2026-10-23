# support-triage: a starter llm-evals suite

Written by `llm-evals init`. It runs as is, offline:

```bash
llm-evals gate . --candidate versions/v2.yaml --provider mock --skip-judges
```

That compares v1 (the baseline) with v2 using the stand-in in `plugins/mock.py`,
so it costs nothing and needs no key. It proves the wiring, not quality.

To make it yours:

1. **Prompt.** Edit `prompts/` and `versions/v1.yaml` (model, params, templates).
2. **Output.** Describe the JSON you expect in `schemas/output.schema.json`.
3. **Cases.** Replace `data/cases.jsonl`: one case per line with an `id`, the
   template variables under `input`, the known answer under `expected`, and
   optional `strata` to slice results by.
4. **Checks.** Built-ins are configured in `suite.yaml`; your own live in
   `plugins/checks.py`. Prefer checks with a known answer over LLM judges.
5. **Run for real.** Set `ANTHROPIC_API_KEY`, then
   `llm-evals estimate .` for the cost and `llm-evals gate . --candidate versions/v2.yaml`.
6. **Measure the noise.** `llm-evals aa . --runs 10` runs the same prompt against
   itself; pick a tolerance (and `samples`) the noise supports.
7. **Gate pull requests.** Add the workflow `llm-evals init --ci` writes, and an
   `ANTHROPIC_API_KEY` repository secret.
