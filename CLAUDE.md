# llm-evals

An eval and regression-gating pipeline for LLM features: paired baseline-vs-candidate
statistics, pluggable checks, and a PROMOTE / HOLD / ROLLBACK / INCOMPLETE gate.

- Module contracts: `docs/contracts.md`. Read it before changing a module boundary.
- Shared types: `src/llm_evals/types.py`. Plug-in registries: `src/llm_evals/registry.py`.
- Run the tests with `uv run pytest`. Tests never call a real model API.
- Requirement ids in comments and docstrings (CAL-4, STAT-9, ...) refer to the original
  design document, which isn't part of this repository.
