"""Known-facts cases: real-world facts the model already knows, contradicted in the corpus.

The research suite's world is fictional, so a model has no prior belief about it
and a changed-fact case can't tell "follows its sources" from "follows memory".
These cases use famous, stable facts instead. Each fact yields two cases:

- `none`: the corpus states the true value; the expected answer is the true value.
- `change_fact`: the corpus states a different value; the expected answer is the
  corpus value, `original_answer` is the true one, and the true value is a
  distractor, so answering from memory fails `answer_correct` and `follows_corpus`.

The changed documents deliberately state false things. Every case records that in
`source.note`, so nobody mistakes these fixtures for a reference source.

Regenerate with:  uv run python -m llm_evals.cases.known_facts suites/research-agent
"""

from __future__ import annotations

import hashlib
import random
import sys
from dataclasses import dataclass
from pathlib import Path

from llm_evals.cases.loader import write_cases
from llm_evals.types import Case


@dataclass(frozen=True)
class KnownFact:
    id: str
    question: str
    title: str
    sentence: str  # "{value}" is replaced with the stated value
    true_value: str
    changed_value: str
    true_aliases: tuple[str, ...] = ()
    changed_aliases: tuple[str, ...] = ()
    filler: str = ""


FACTS: tuple[KnownFact, ...] = (
    KnownFact("eiffel-opened", "In what year did the Eiffel Tower open?", "The Eiffel Tower",
              "The Eiffel Tower in Paris opened in {value}.", "1889", "1893",
              filler="The wrought-iron lattice tower stands on the Champ de Mars and is one of the most visited monuments in the world."),
    KnownFact("berlin-wall-fell", "In what year did the Berlin Wall fall?", "The Berlin Wall",
              "The Berlin Wall fell in {value}, when the border crossings were opened.", "1989", "1991",
              filler="The wall divided East and West Berlin, and parts of it are preserved as memorials."),
    KnownFact("australia-capital", "What is the capital city of Australia?", "Australia",
              "The capital city of Australia is {value}.", "Canberra", "Melbourne",
              filler="Australia is a federation of six states and two mainland territories."),
    KnownFact("gold-atomic-number", "What is the atomic number of gold?", "Gold",
              "Gold is a chemical element with the symbol Au and atomic number {value}.", "79", "81",
              filler="It is a dense, soft, yellow metal that has been used for coinage and jewellery for millennia."),
    KnownFact("apollo-11-landing", "In what year did Apollo 11 land on the Moon?", "Apollo 11",
              "Apollo 11 landed on the Moon in {value}, and Neil Armstrong became the first person to walk on its surface.",
              "1969", "1971", filler="The mission launched from Kennedy Space Center in Florida."),
    KnownFact("canada-capital", "What is the capital city of Canada?", "Canada",
              "The capital city of Canada is {value}.", "Ottawa", "Toronto",
              filler="Canada has ten provinces and three territories."),
    KnownFact("pride-and-prejudice-author", "Who wrote the novel Pride and Prejudice?", "Pride and Prejudice",
              "Pride and Prejudice is a novel by {value}, first published in 1813.", "Jane Austen",
              "Charlotte Bronte", true_aliases=("Austen",), changed_aliases=("Bronte", "Charlotte Brontë"),
              filler="The novel follows Elizabeth Bennet and Fitzwilliam Darcy."),
    KnownFact("titanic-sank", "In what year did the Titanic sink?", "RMS Titanic",
              "RMS Titanic sank in the North Atlantic in {value} after striking an iceberg.", "1912", "1914",
              filler="The ship was on its maiden voyage from Southampton to New York."),
    KnownFact("japan-capital", "What is the capital city of Japan?", "Japan",
              "The capital city of Japan is {value}.", "Tokyo", "Kyoto",
              filler="Japan is an island country in East Asia made up of four main islands and many smaller ones."),
    KnownFact("mona-lisa-painter", "Who painted the Mona Lisa?", "The Mona Lisa",
              "The Mona Lisa is a portrait painted by {value}.", "Leonardo da Vinci", "Raphael",
              true_aliases=("Leonardo", "da Vinci"), changed_aliases=("Raffaello Sanzio",),
              filler="It hangs in the Louvre in Paris."),
    KnownFact("ww2-ended", "In what year did World War II end?", "World War II",
              "World War II ended in {value}.", "1945", "1946",
              filler="The war involved most of the world's nations and was the deadliest conflict in history."),
    KnownFact("first-iphone", "In what year was the first iPhone released?", "The iPhone",
              "Apple released the first iPhone in {value}.", "2007", "2009",
              filler="The phone combined a music player, a mobile phone and an internet browser."),
    KnownFact("brazil-capital", "What is the capital city of Brazil?", "Brazil",
              "The capital city of Brazil is {value}.", "Brasília", "Rio de Janeiro",
              true_aliases=("Brasilia",), changed_aliases=("Rio",),
              filler="Brazil is the largest country in South America."),
    KnownFact("declaration-signed", "In what year was the United States Declaration of Independence signed?",
              "The Declaration of Independence",
              "The United States Declaration of Independence was signed in {value}.", "1776", "1781",
              filler="The document was adopted by the Second Continental Congress in Philadelphia."),
    KnownFact("oxygen-atomic-number", "What is the atomic number of oxygen?", "Oxygen",
              "Oxygen is a chemical element with the symbol O and atomic number {value}.", "8", "10",
              filler="It makes up about a fifth of the Earth's atmosphere."),
    KnownFact("hamlet-author", "Who wrote the play Hamlet?", "Hamlet",
              "Hamlet is a tragedy written by {value}.", "William Shakespeare", "Christopher Marlowe",
              true_aliases=("Shakespeare",), changed_aliases=("Marlowe",),
              filler="The play is set in Denmark and follows Prince Hamlet."),
)

NOTE = ("Known-facts test fixture. In change_fact cases the document deliberately states a false "
        "value to test whether the agent follows its sources; do not use it as a reference.")


def _hash(*parts: object, n: int = 10) -> str:
    return hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:n]


def _doc(fact: KnownFact, value: str, variant: str) -> dict[str, str]:
    text = f"{fact.sentence.format(value=value)} {fact.filler}".strip()
    return {"id": f"kf-{_hash(fact.id, variant, value, n=6)}", "title": fact.title, "text": text}


def build_cases(seed: int = 7, n_distractors: int = 5) -> list[Case]:
    rng = random.Random(seed)
    true_docs = {f.id: _doc(f, f.true_value, "true") for f in FACTS}
    cases: list[Case] = []
    for fact in FACTS:
        others = [true_docs[f.id] for f in FACTS if f.id != fact.id]
        distractors = rng.sample(others, n_distractors)
        for op in ("none", "change_fact"):
            changed = op == "change_fact"
            value = fact.changed_value if changed else fact.true_value
            main = _doc(fact, value, "changed" if changed else "true")
            docs = [main, *distractors]
            rng.shuffle(docs)
            expected = {
                "behavior": "answer",
                "answer": value,
                "aliases": list(fact.changed_aliases if changed else fact.true_aliases),
                "original_answer": fact.true_value if changed else None,
                "distractors": [fact.true_value, *fact.true_aliases] if changed else [],
            }
            perturbation = {"op": op}
            if changed:
                perturbation |= {"fact_id": fact.id, "old_value": fact.true_value,
                                 "new_value": fact.changed_value, "doc_ids": [main["id"]]}
            cases.append(Case(
                id=f"kf-{fact.id}" + ("-change" if changed else ""),
                input={"question": fact.question},
                expected=expected,
                fixtures={"corpus_id": f"kf-corpus-{_hash(fact.id, op)}", "docs": docs},
                source={"facts": [{"id": fact.id, "subject": fact.title, "value": value,
                                   "true_value": fact.true_value, "doc_ids": [main["id"]]}],
                        "perturbation": perturbation, "note": NOTE},
                strata={"qtype": "known_fact", "perturbation": op, "behavior": "answer"},
                tags=["known_fact", op],
                provenance="public",
            ))
    return cases


def build_known_fact_cases(suite_dir: Path, seed: int = 7) -> Path:
    return write_cases(Path(suite_dir) / "data" / "known_facts_cases.jsonl", build_cases(seed))


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    path = build_known_fact_cases(Path(args[0] if args else "suites/research-agent"))
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
