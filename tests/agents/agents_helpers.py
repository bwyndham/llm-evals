"""Shared test data for tests/agents."""

from __future__ import annotations

import json
from typing import Any

from llm_evals.types import Case

# Fixture prompts: a baseline and the three seeded variants, each dropping one sentence.
DOCS_ONLY = "Answer only from the documents you fetch; do not use outside knowledge."
CITE = "Every claim must cite the doc_id of each document that supports it."
ABSTAIN = 'If the documents do not contain the answer, set status to "not_found" and leave answer empty.'
CONFLICTS = "If sources conflict, list each conflict in conflicts and leave answer empty."
SCHEMA = (
    "Reply with only this JSON:\n"
    "```json\n"
    '{"status": "answered" | "not_found", "answer": "...", '
    '"claims": [{"text": "...", "citations": ["doc_id"]}], "conflicts": ["..."]}\n'
    "```\n"
)


def system_prompt(*, docs_only: bool = True, cite: bool = True, abstain: bool = True) -> str:
    parts = ["You are a research assistant. Use the search and fetch tools to find the answer."]
    if docs_only:
        parts.append(DOCS_ONLY)
    if cite:
        parts.append(CITE)
    if abstain:
        parts.append(ABSTAIN)
    parts.append(CONFLICTS)
    return " ".join(parts) + "\n\n" + SCHEMA


VARIANTS = {
    "baseline": {},
    "ignore_corpus": {"docs_only": False},
    "drop_citations": {"cite": False},
    "no_abstain": {"abstain": False},
}

QUESTION = "In what year was the Halvorsen Bridge completed?"

DOCS: list[dict[str, str]] = [
    {
        "id": "d1",
        "title": "Halvorsen Bridge",
        "text": (
            "The Halvorsen Bridge crosses the Vell River in Ostmark. "
            "The Halvorsen Bridge was completed in 1931 after six years of work. "
            "It is 412 metres long and carries two lanes of traffic."
        ),
    },
    {
        "id": "d2",
        "title": "Vell River",
        "text": (
            "The Vell River runs 240 kilometres through Ostmark. "
            "Several bridges cross it, including the Halvorsen Bridge."
        ),
    },
    {
        "id": "d3",
        "title": "Ostmark transport history",
        "text": "Ostmark opened its first railway in 1874. The Kessel Tunnel was completed in 1952.",
    },
    {
        "id": "d4",
        "title": "Ostmark harbour",
        "text": "The harbour handles fishing boats and a summer ferry to the islands.",
    },
]


def make_case(docs: list[dict[str, str]] | None = None, *, question: str = QUESTION,
              case_id: str = "c1", op: str = "none") -> Case:
    return Case(
        id=case_id,
        input={"question": question},
        fixtures={"corpus_id": "ostmark", "docs": docs if docs is not None else DOCS},
        expected={"behavior": "answer", "answer": "1931", "aliases": [], "original_answer": None},
        source={"facts": [], "perturbation": {"op": op}},
        provenance="constructed",
    )


GOOD_OUTPUT = {
    "status": "answered",
    "answer": "1931",
    "claims": [{"text": "The Halvorsen Bridge was completed in 1931.", "citations": ["d1"]}],
    "conflicts": [],
}
