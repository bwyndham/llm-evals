# Spec-writing suite

Turns a messy product ask (meeting notes, Slack threads, emails) into a ticket with a
title, a description, Gherkin acceptance criteria, assumptions and open questions.

- **System under test:** `claude-haiku-5-5` with the prompt named in `versions/`. The
  current baseline is `versions/v2.yaml`.
- **Cases:** 120 synthetic asks rendered from known requirement atoms
  (`data/cases.jsonl`), so every case has a true answer.
- **Gate:** the deterministic checks in `suite.yaml` decide PROMOTE, HOLD or ROLLBACK
  at a 5-point tolerance. The LLM judges (faithfulness, checklist, pairwise) are
  reported but don't gate until each is calibrated against human labels.

A PR that changes this suite runs the gate in CI (`.github/workflows/evals.yml`) and
posts the result as a PR comment. To try a prompt change, edit the baseline manifest's
prompt in the PR: CI compares it with the base branch's version.

Human labels for judge calibration are backed up in `data/labels/` (one JSONL file per
judge), each with the exact output it judged. Refresh the file after labeling with
`eval calibration backup-labels --db .evals/evals.sqlite --out data/labels/faithfulness.jsonl`
(run from this folder). On a fresh machine, `eval calibration restore-labels` rebuilds the
labeled samples in a new run, ready to re-judge.
