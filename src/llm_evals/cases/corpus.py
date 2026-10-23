"""Synthetic corpora, questions and corpus perturbations for the research-agent suite.

    uv run python -m llm_evals.cases.corpus suites/research-agent

Everything here is invented: the Republic of Vellmora, an island nation with its
towns, companies, people, provinces and landmarks. `data/world_facts.jsonl` holds
the atomic facts; it doubles as the "parametric memory" of the offline mock
model, so its values must never change.

From those facts this module builds about 35 short documents (deterministic for a
seed), then three kinds of question:

- `lookup`: one fact ("What is the population of Drennick?")
- `two_hop`: two facts in different documents ("What is the population of the town
  where Kestrin Lamp Company is headquartered?")
- `comparison`: two facts with the same attribute ("Which was founded earlier, ...?")

Each case gets its own corpus of 8-12 documents: the documents that hold the
answer, look-alike distractors (New Drennick for Drennick, Kestrin Lantern Works
for Kestrin Lamp Company, ...) and random filler documents.

CASE-6 perturbations each turn a base case into a new case with an id suffix and
`source.perturbation` describing what changed:

- `plant_fact` (`-plant`): add a document with a fact that is not in the world
  facts and ask about it. Behavior `answer`.
- `remove_answer_doc` (`-remove`): drop every document that mentions the answer
  (or an alias). Behavior `abstain`, answer None.
- `change_fact` (`-change`): rewrite the fact's value in every document. Behavior
  `answer` with the new value; `original_answer` is the old (memorised) answer.
- `add_conflict` (`-conflict`): add a document that states a different value.
  Behavior `flag_conflict`, answer None; both values are in `source.perturbation`.

The answer helpers at the top (`normalize_answer`, `answer_matches`,
`text_mentions`) are imported by the research checks; keep their signatures.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import sys
import unicodedata
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from llm_evals.cases.loader import case_set_hash, load_cases, write_cases
from llm_evals.types import Case

WORLD_FACTS_FILE = Path("data/world_facts.jsonl")
CASES_FILE = Path("data/cases.jsonl")
CORPUS_MIN = 8
CORPUS_MAX = 12
DOC_MIN_WORDS = 100
DOC_MAX_WORDS = 300
YEAR_MIN, YEAR_MAX = 1650, 2024

QType = Literal["lookup", "two_hop", "comparison"]
Op = Literal["none", "plant_fact", "remove_answer_doc", "change_fact", "add_conflict"]
QTYPES: tuple[str, ...] = ("lookup", "two_hop", "comparison")
OPS: tuple[str, ...] = ("none", "plant_fact", "remove_answer_doc", "change_fact", "add_conflict")
OP_SUFFIX = {
    "plant_fact": "plant",
    "remove_answer_doc": "remove",
    "change_fact": "change",
    "add_conflict": "conflict",
}
BEHAVIOR = {
    "none": "answer",
    "plant_fact": "answer",
    "remove_answer_doc": "abstain",
    "change_fact": "answer",
    "add_conflict": "flag_conflict",
}


# =============================================================================
# Answer normalisation and matching (stable API, used by checks/research.py)
# =============================================================================

_UNIT_PHRASES = [
    (r"\bsquare kilomet(?:re|er)s?\b", "km2"),
    (r"\bsq\.? ?km\b", "km2"),
    (r"\bkm\^?2\b", "km2"),
    (r"\bkilomet(?:re|er)s?\b", "km"),
    (r"\bkms\b", "km"),
    (r"\bmet(?:re|er)s?\b", "m"),
    (r"\bper ?cent\b", "%"),
]
UNIT_TOKENS = frozenset({"km", "km2", "m", "%", "cm", "mm", "kg", "t"})
_LEADING_WORDS = frozenset({"the", "a", "an", "mayor", "dr", "mr", "mrs", "ms", "sir", "dame"})
_NUMBER_TOKEN = re.compile(r"\d+(?:\.\d+)?")


def _normalize_tokens(text: str) -> list[str]:
    s = unicodedata.normalize("NFKC", text).lower()  # NFKC turns "km²" into "km2"
    s = re.sub(r"[‐-―]", "-", s)
    # thousands separators: "48,200" / "48 200" (nbsp, thin space) -> "48200"
    prev = None
    while prev != s:
        prev = s
        s = re.sub(r"(?<=\d)[,   ](?=\d{3}(?!\d))", "", s)
    s = re.sub(r"(\d)([a-z%])", r"\1 \2", s)  # "84km" -> "84 km"
    s = re.sub(r"(?<!\d)\.|\.(?!\d)", " ", s)  # keep only decimal points
    s = re.sub(r"[^a-z0-9.%\s]", " ", s)
    for pattern, repl in _UNIT_PHRASES:
        s = re.sub(pattern, repl, s)
    tokens = []
    for tok in s.split():
        if _NUMBER_TOKEN.fullmatch(tok) and "." in tok:
            tok = tok.rstrip("0").rstrip(".")  # "12.50" -> "12.5", "1887.0" -> "1887"
        tokens.append(tok)
    return tokens


def normalize_answer(text: str | int | float | None) -> str:
    """Canonical form of a short answer, for comparing answers.

    Lowercases; NFKC; drops thousands separators ("48,200" -> "48200"); strips
    trailing decimal zeros ("12.50" -> "12.5"); maps unit words to symbols
    ("kilometres" -> "km", "square kilometres"/"km²" -> "km2", "metres" -> "m",
    "per cent" -> "%"); splits "84km" into "84 km"; turns punctuation and hyphens
    into spaces; drops leading articles and titles ("the", "a", "mayor", "dr", ...);
    collapses whitespace. None gives "".
    """
    if text is None:
        return ""
    tokens = _normalize_tokens(str(text))
    while tokens and tokens[0] in _LEADING_WORDS and len(tokens) > 1:
        tokens = tokens[1:]
    return " ".join(tokens)


def _find_sublist(haystack: Sequence[str], needle: Sequence[str]) -> list[int]:
    n = len(needle)
    if n == 0:
        return []
    return [i for i in range(len(haystack) - n + 1) if list(haystack[i : i + n]) == list(needle)]


def answer_matches(
    output_answer: str | None,
    expected: str | Mapping[str, Any] | None,
    aliases: Iterable[str] = (),
    *,
    distractors: Iterable[str] = (),
) -> bool:
    """True if `output_answer` gives the expected answer; answers may be sentences.

    `expected` is either the answer string or a case's `expected` dict, whose
    `answer`, `aliases` and `distractors` are used (`original_answer` is ignored:
    call `answer_matches(out, case.expected["original_answer"])` to detect a
    memorised answer). Extra `aliases` / `distractors` are added to the dict's.
    A None expected answer (abstain / flag_conflict cases) never matches.

    Both sides go through `normalize_answer`, so numbers compare by value
    ("12,500" == "12500" == "12500.0") and units by symbol. The answer matches
    when the expected answer or any alias appears in it as a whole token run
    ("The town was founded in 1847." matches "1847"; "12,500 people" matches
    "12500"), except that:
    - a number followed directly by a different unit doesn't match ("84 m" is
      not "84 km"); a "<number> <unit>" candidate also matches the bare number;
    - if any distractor appears in the answer, it doesn't match unless the answer
      equals a candidate exactly. Cases list look-alike entities and the losing
      side of a comparison as `distractors`, so "New Drennick" is not "Drennick"
      and "Thornquay is larger than Velmouth" is not "Velmouth".
    """
    if isinstance(expected, Mapping):
        aliases = [*(expected.get("aliases") or []), *aliases]
        distractors = [*(expected.get("distractors") or []), *distractors]
        expected = expected.get("answer")
    if output_answer is None or expected is None:
        return False
    out = normalize_answer(output_answer).split()
    if not out:
        return False
    cands = [c for c in (normalize_answer(x).split() for x in [expected, *aliases]) if c]
    if any(out == c for c in cands):
        return True
    for d in distractors:
        dt = normalize_answer(d).split()
        if not dt or any(_find_sublist(c, dt) for c in cands):
            continue  # a distractor inside the answer itself ("Kestrin" in "Kestrin Lamp Company")
        if _find_sublist(out, dt):
            return False
    for c in cands:
        runs = [c]
        if len(c) == 2 and _NUMBER_TOKEN.fullmatch(c[0]) and c[1] in UNIT_TOKENS:
            runs.append(c[:1])  # "84 km" also matches a bare "84"
        for run in runs:
            for idx in _find_sublist(out, run):
                end = idx + len(run)
                if _NUMBER_TOKEN.fullmatch(run[-1]) and end < len(out) and out[end] in UNIT_TOKENS:
                    continue  # "84" followed by a unit the candidate doesn't name
                return True
    return False


def text_mentions(text: str, value: str, aliases: Iterable[str] = ()) -> bool:
    """True if `text` mentions `value` or any alias as a whole token run (normalised).

    Uses the same normalisation as `normalize_answer` (so "48,200" in a document
    matches the alias "48200", and "Drennick" matches inside "New Drennick").
    """
    tokens = _normalize_tokens(text)
    for cand in [value, *aliases]:
        c = _normalize_tokens(cand)
        if c and _find_sublist(tokens, c):
            return True
    return False


# =============================================================================
# The world: facts, entities, documents
# =============================================================================


class WorldFact(BaseModel):
    """One line of data/world_facts.jsonl."""

    id: str
    subject: str
    attribute: str
    value: str
    aliases: list[str] = Field(default_factory=list)
    sentence: str


YEAR_ATTRS = frozenset({"chartered", "founded", "launched", "born", "opened", "first_held"})
COUNT_ATTRS = frozenset({"population", "employees"})
MEASURE_ATTRS = frozenset({"road_distance_from_port_ilvane", "area", "height"})
ENTITY_ATTRS = frozenset(
    {"mayor", "province", "headquarters", "ceo", "founder", "flagship_product", "capital"}
)


def value_kind(attribute: str) -> Literal["year", "count", "measure", "entity"]:
    if attribute in YEAR_ATTRS:
        return "year"
    if attribute in COUNT_ATTRS:
        return "count"
    if attribute in MEASURE_ATTRS:
        return "measure"
    return "entity"


def is_numeric(attribute: str) -> bool:
    return value_kind(attribute) != "entity"


# Noun phrase for a subject when it appears inside a question or filler text.
DISPLAY = {
    "Greywater Bridge": "the Greywater Bridge",
    "Greywater Viaduct": "the Greywater Viaduct",
    "Coastline Railway": "the Coastline Railway",
    "Ilvane Lantern Festival": "the Ilvane Lantern Festival",
    "Gullwing Lantern": "the Gullwing Lantern",
    "Tidewire Broadband": "the Tidewire Broadband service",
    "Sunbank Battery": "the Sunbank Battery",
    "Velmouth Ferry Society": "the Velmouth Ferry Society",
    "Saltreed Lighthouse": "the Saltreed Lighthouse",
    "Marram Star": "the Marram Star",
}

# Aliases accepted for an entity named as an answer (comparison winners).
ENTITY_ALIASES = {
    "Kestrin Lamp Company": ["Kestrin Lamp Co.", "Kestrin Lamp"],
    "Velmouth Ferryworks": ["Velmouth Ferry Works"],
    "Brindlecoast Telecom": ["Brindlecoast"],
    "Orsk Mineral Partners": ["Orsk Mineral"],
    "Halvane Biscuit Company": ["Halvane Biscuit Co.", "Halvane Biscuits"],
    "Ilvane Solar Cooperative": ["Ilvane Solar", "Ilvane Solar Co-op"],
    "Harrowmark Province": ["Harrowmark"],
    "Lissenford Province": ["Lissenford"],
    "Ostervale Province": ["Ostervale"],
    "Gullwing Lantern": ["Gullwing"],
    "Tidewire Broadband": ["Tidewire"],
    "Sunbank Battery": ["Sunbank"],
    "Casimir Holloway": ["Holloway"],
    "Odalys Ferrant": ["Ferrant"],
    "Augustin Merle": ["Merle"],
    "Ilse Varga": ["Varga"],
}

# Look-alike entities: their documents are added to a case as distractors.
LOOKALIKES = {
    "Port Ilvane": ["Ilvane Junction"],
    "Ilvane Junction": ["Port Ilvane"],
    "Drennick": ["New Drennick"],
    "New Drennick": ["Drennick"],
    "Kestrin": ["Upper Kestrin", "Kestrin Lantern Works"],
    "Upper Kestrin": ["Kestrin"],
    "Kestrin Lamp Company": ["Kestrin Lantern Works", "Kestrin"],
    "Kestrin Lantern Works": ["Kestrin Lamp Company"],
    "Velmouth Ferryworks": ["Velmouth Ferry Society"],
    "Velmouth Ferry Society": ["Velmouth Ferryworks"],
    "Mount Caddow": ["Caddow Hill"],
    "Caddow Hill": ["Mount Caddow"],
    "Greywater Bridge": ["Greywater Viaduct"],
    "Greywater Viaduct": ["Greywater Bridge"],
    "Gullwing Lantern": ["Kestrin Lantern Works"],
}


def np(subject: str) -> str:
    """The noun phrase used for `subject` in questions ("the Greywater Bridge")."""
    return DISPLAY.get(subject, subject)


def _cap(text: str) -> str:
    return text[:1].upper() + text[1:]


# (slug, genre, subject, title, fact ids). Fact sentences are used verbatim;
# filler sentences come from FILLER[genre] and never contain digits or facts.
DOC_SPECS: list[tuple[str, str, str, str, list[str]]] = [
    ("port-ilvane", "town", "Port Ilvane", "Port Ilvane: a town profile",
     ["port-ilvane-population", "port-ilvane-chartered", "port-ilvane-mayor", "port-ilvane-province"]),
    ("drennick", "town", "Drennick", "Drennick at a glance",
     ["drennick-population", "drennick-chartered", "drennick-mayor", "drennick-province"]),
    ("thornquay", "town", "Thornquay", "Thornquay, harbour town of the east coast",
     ["thornquay-population", "thornquay-chartered", "thornquay-mayor", "thornquay-province"]),
    ("kestrin", "town", "Kestrin", "Kestrin: a town profile", ["kestrin-population"]),
    ("velmouth", "town", "Velmouth", "Velmouth, the ferry town",
     ["velmouth-population", "velmouth-mayor", "velmouth-province"]),
    ("orsk-harbour", "town", "Orsk Harbour", "Orsk Harbour: a town profile", ["orsk-harbour-population"]),
    ("ambery-falls", "town", "Ambery Falls", "Ambery Falls: a town profile", ["ambery-falls-population"]),
    ("ilvane-junction", "town", "Ilvane Junction", "Ilvane Junction, a railway village",
     ["ilvane-junction-population"]),
    ("new-drennick", "town", "New Drennick", "New Drennick: a town profile",
     ["new-drennick-population", "new-drennick-chartered"]),
    ("upper-kestrin", "town", "Upper Kestrin", "Upper Kestrin: a village profile", ["upper-kestrin-population"]),
    ("road-guide", "guide", "Vellmora", "Driving in Vellmora: distances from the capital",
     ["drennick-road-distance", "thornquay-road-distance"]),
    ("kestrin-lamp-company", "company", "Kestrin Lamp Company", "Kestrin Lamp Company: company profile",
     ["kestrin-lamp-company-founded", "kestrin-lamp-company-headquarters", "kestrin-lamp-company-ceo",
      "kestrin-lamp-company-founder", "kestrin-lamp-company-employees", "kestrin-lamp-company-flagship"]),
    ("velmouth-ferryworks", "company", "Velmouth Ferryworks", "Velmouth Ferryworks: company profile",
     ["velmouth-ferryworks-founded", "velmouth-ferryworks-headquarters", "velmouth-ferryworks-employees"]),
    ("brindlecoast-telecom", "company", "Brindlecoast Telecom", "Brindlecoast Telecom: company profile",
     ["brindlecoast-telecom-founded", "brindlecoast-telecom-headquarters", "brindlecoast-telecom-ceo",
      "brindlecoast-telecom-employees", "brindlecoast-telecom-flagship"]),
    ("orsk-mineral-partners", "company", "Orsk Mineral Partners", "Orsk Mineral Partners: company profile",
     ["orsk-mineral-partners-founded", "orsk-mineral-partners-headquarters", "orsk-mineral-partners-employees"]),
    ("halvane-biscuit-company", "company", "Halvane Biscuit Company", "Halvane Biscuit Company: company profile",
     ["halvane-biscuit-company-founded", "halvane-biscuit-company-headquarters",
      "halvane-biscuit-company-founder", "halvane-biscuit-company-employees"]),
    ("ilvane-solar-cooperative", "company", "Ilvane Solar Cooperative",
     "Ilvane Solar Cooperative: company profile",
     ["ilvane-solar-cooperative-founded", "ilvane-solar-cooperative-headquarters",
      "ilvane-solar-cooperative-flagship"]),
    ("kestrin-lantern-works", "company", "Kestrin Lantern Works", "Kestrin Lantern Works: company profile",
     ["kestrin-lantern-works-founded", "kestrin-lantern-works-headquarters", "kestrin-lantern-works-ceo"]),
    ("velmouth-ferry-society", "society", "Velmouth Ferry Society", "The Velmouth Ferry Society",
     ["velmouth-ferry-society-founded"]),
    ("gullwing-lantern", "product", "Gullwing Lantern", "The Gullwing Lantern", ["gullwing-lantern-launched"]),
    ("tidewire-broadband", "product", "Tidewire Broadband", "Tidewire Broadband",
     ["tidewire-broadband-launched"]),
    ("sunbank-battery", "product", "Sunbank Battery", "The Sunbank Battery", ["sunbank-battery-launched"]),
    ("casimir-holloway", "person", "Casimir Holloway", "Casimir Holloway: a short biography",
     ["casimir-holloway-born"]),
    ("odalys-ferrant", "person", "Odalys Ferrant", "Odalys Ferrant: a short biography", ["odalys-ferrant-born"]),
    ("augustin-merle", "person", "Augustin Merle", "Augustin Merle: a short biography", ["augustin-merle-born"]),
    ("ilse-varga", "person", "Ilse Varga", "Ilse Varga: a short biography", ["ilse-varga-born"]),
    ("harrowmark-province", "province", "Harrowmark Province", "Harrowmark Province",
     ["harrowmark-province-capital", "harrowmark-province-area"]),
    ("lissenford-province", "province", "Lissenford Province", "Lissenford Province",
     ["lissenford-province-capital", "lissenford-province-area"]),
    ("ostervale-province", "province", "Ostervale Province", "Ostervale Province",
     ["ostervale-province-capital", "ostervale-province-area"]),
    ("mount-caddow", "landmark", "Mount Caddow", "Mount Caddow", ["mount-caddow-height"]),
    ("caddow-hill", "landmark", "Caddow Hill", "Caddow Hill", ["caddow-hill-height"]),
    ("greywater-bridge", "landmark", "Greywater Bridge", "The Greywater Bridge", ["greywater-bridge-opened"]),
    ("greywater-viaduct", "landmark", "Greywater Viaduct", "The Greywater Viaduct",
     ["greywater-viaduct-opened"]),
    ("coastline-railway", "landmark", "Coastline Railway", "The Coastline Railway", ["coastline-railway-opened"]),
    ("ilvane-lantern-festival", "event", "Ilvane Lantern Festival", "The Ilvane Lantern Festival",
     ["ilvane-lantern-festival-first-held"]),
]

INTRO = {
    "town": "This profile of {name} is part of a series on the towns and villages of Vellmora, an island republic in the northern sea.",
    "company": "{name} is one of the businesses covered in this series of company profiles from across the island republic of Vellmora.",
    "person": "This short biography of {name} is drawn from public records and local press coverage in Vellmora.",
    "product": "This entry describes {name}, a product designed, made and sold in the island republic of Vellmora.",
    "province": "{name} is one of the provinces of the Republic of Vellmora, an island nation in the northern sea.",
    "landmark": "{name} is one of the places listed in the Vellmora heritage register.",
    "event": "{name} is one of the best-known events in the Vellmora calendar.",
    "society": "{name} is a volunteer heritage group in the island republic of Vellmora.",
    "guide": "This guide summarises driving routes across {name} for visitors and new residents.",
}

FILLER = {
    "town": [
        "{name} sits on a stretch of coast known for its sudden fogs and bright, windy afternoons.",
        "A weekly market fills the main square of {name}, with stalls selling smoked fish, wool and garden produce.",
        "Most visitors arrive in {name} by bus or ferry and explore the older streets on foot.",
        "The council of {name} meets in a stone hall that was restored after a long public campaign.",
        "Local walking groups maintain a network of footpaths that lead out of {name} towards the headlands.",
        "Residents describe a quiet place in winter that grows lively once the summer festivals begin.",
        "The parish church in {name} is known for its painted ceiling and its unusually loud bell.",
        "Schools in {name} share a sports ground with the neighbouring villages.",
        "Fishing remains part of daily life here, although tourism and small workshops now provide many jobs.",
        "A small museum in {name} keeps maps, photographs and letters donated by local families.",
        "Cafes along the waterfront stay open late during the summer months.",
    ],
    "company": [
        "Staff at {name} describe a practical, close-knit culture in which managers still walk the shop floor.",
        "{name} sells most of its output at home, though it has slowly built a following overseas.",
        "Trade unions and the management of {name} negotiate a new agreement every few years.",
        "The company publishes a short annual review, which local newspapers tend to quote at length.",
        "{name} sponsors a youth football league and a winter lecture series.",
        "Analysts who follow {name} point to steady demand and careful spending as its main strengths.",
        "Visitors can tour parts of the premises on selected weekends.",
        "{name} has invested in training programmes run jointly with the national technical college.",
        "Its suppliers are mostly small firms based along the coast.",
        "The board meets four times a year and publishes brief minutes of each meeting.",
        "Customers often praise the company for answering letters and calls promptly.",
    ],
    "person": [
        "{short} grew up in a household where reading aloud after dinner was a family habit.",
        "Colleagues remember {short} as patient, exacting and quick to share credit.",
        "{short} has often spoken about the value of apprenticeships and practical training.",
        "Outside work, {short} is remembered as a keen sailor and an occasional painter of seascapes.",
        "Several letters written by {short} are now held in the national archive.",
        "{short} served on the boards of a number of local charities.",
        "Interviewers often noted that {short} preferred to talk about colleagues rather than personal achievements.",
        "{short} travelled widely but always kept a home on the island.",
        "Friends describe a dry sense of humour and a fondness for long coastal walks.",
        "A portrait of {short} hangs in the reading room of the town library.",
    ],
    "product": [
        "Early reviews of {name} praised its simple design and its durability.",
        "{name} is sold through hardware shops, post offices and an online store.",
        "Owners have formed an informal club that swaps repair tips and spare parts.",
        "The design has been updated several times without changing its basic shape.",
        "Advertising for {name} has long featured photographs of the island's coastline.",
        "Retailers report that it sells steadily throughout the year rather than in seasonal bursts.",
        "A small team of engineers continues to look after {name} and its accessories.",
        "Collectors look out for early examples, which turn up now and then at country auctions.",
        "Spare parts remain easy to find, which owners count as one of its virtues.",
        "Its packaging carries a short note on recycling and repair.",
    ],
    "province": [
        "The landscape of {name} mixes rolling farmland, sheltered bays and patches of old woodland.",
        "Regional government is handled by an elected assembly that meets in the provincial capital.",
        "Roads are well maintained, though some upland routes close during winter storms.",
        "Agriculture, fishing and tourism are the main sources of work in {name}.",
        "{name} hosts a series of agricultural shows each summer.",
        "Bird watchers visit for the seabird colonies on its cliffs.",
        "Its villages are known for slate roofs and brightly painted doors.",
        "A network of bus routes links the villages of {name} with the main towns.",
        "Several nature reserves protect the wetlands along its rivers.",
        "Local dialects in {name} keep many words borrowed from old seafaring speech.",
    ],
    "landmark": [
        "Guided walks to {name} leave from the nearest town on most weekends.",
        "Photographers favour {name} in the early morning, when the light is low and the paths are empty.",
        "A small information board explains its history and the local wildlife.",
        "{name} appears on postcards, tea towels and the cover of at least one popular guidebook.",
        "Conservation volunteers inspect {name} regularly and report any damage to the heritage office.",
        "Access can be restricted during severe weather.",
        "Local stories about {name} are collected in a booklet sold at the visitor centre.",
        "Painters have been drawn to the spot for generations.",
        "A path suitable for wheelchairs now leads to a viewing point nearby.",
        "School groups often visit as part of their local history lessons.",
    ],
    "event": [
        "During {name}, streets are hung with paper lanterns and shops stay open late.",
        "Volunteers spend months preparing, building floats and rehearsing music.",
        "Families travel from across the island to attend {name}.",
        "The programme usually includes concerts, a parade and a night-time procession to the harbour.",
        "Organisers rely on donations and a small grant from the town council.",
        "Photographs of {name} are displayed each winter in the town library.",
        "Food stalls serve fried fish, spiced bread and hot cider until late in the evening.",
        "Local schools make lanterns for the procession in the weeks before it begins.",
        "Visitors are advised to book rooms well in advance.",
    ],
    "society": [
        "Members of {name} restore old vessels and record the memories of retired crews.",
        "{name} publishes a newsletter and runs open days at the harbour.",
        "Membership is open to anyone with an interest in local maritime history.",
        "The group has raised funds through raffles, concerts and the sale of calendars.",
        "Its archive includes logbooks, tickets and timetables collected over many decades.",
        "Volunteers lead harbour walks on summer evenings.",
        "{name} works closely with the town museum on its exhibitions.",
        "New members are welcomed at a supper held each autumn.",
        "Its workshop is open to visitors on the first weekend of each month.",
    ],
    "guide": [
        "Most main roads on the island are two lanes wide and follow the coast.",
        "Drivers should allow extra time in summer, when farm traffic and tourist coaches share the roads.",
        "Fuel stations are spread fairly evenly, though some inland stretches have none.",
        "Ferries link the main island with the smaller isles, and timetables change with the seasons.",
        "Speed cameras operate near schools and in the centre of every town.",
        "Winter storms occasionally close exposed coastal sections for a few hours at a time.",
        "Parking in town centres is usually limited to short stays during the day.",
        "Cyclists share many rural roads, so drivers should take care on bends.",
        "Road signs are written in plain block capitals and are easy to read.",
    ],
}

CONFLICT_FRAMES = [
    ("{name}: entry from the Coastal Almanac",
     "The following note on {name} is taken from the Coastal Almanac, a reference book compiled by volunteer correspondents across Vellmora."),
    ("{name}: notes from a regional newsletter",
     "A regional newsletter recently ran a short feature on {name}, summarised below for readers of this collection."),
    ("{name}: visitor leaflet",
     "This leaflet about {name} was produced for visitors by a local tourism association in Vellmora."),
]


@dataclass(frozen=True)
class PlantedFact:
    """A fact that exists only in a planted document, never in the world facts."""

    id: str
    subject: str
    attribute: str
    value: str
    aliases: tuple[str, ...]
    sentence: str
    question: str
    title: str
    genre: str

    def as_source(self, doc_ids: list[str]) -> dict[str, Any]:
        return {"id": self.id, "subject": self.subject, "attribute": self.attribute,
                "value": self.value, "aliases": list(self.aliases), "doc_ids": doc_ids}


PLANT_FACTS: list[PlantedFact] = [
    PlantedFact("plant-saltreed-lighthouse-first-lit", "Saltreed Lighthouse", "first_lit", "1908", (),
                "The Saltreed Lighthouse, on the headland north of Orsk Harbour, was first lit in 1908.",
                "In what year was the Saltreed Lighthouse first lit?", "The Saltreed Lighthouse", "landmark"),
    PlantedFact("plant-marram-star-passengers", "Marram Star", "passenger_capacity", "340", (),
                "The Velmouth Ferryworks vessel Marram Star is licensed to carry 340 passengers.",
                "How many passengers is the Marram Star licensed to carry?", "The Marram Star", "product"),
    PlantedFact("plant-kestrin-lamp-company-mascot", "Kestrin Lamp Company", "mascot", "Pell", (),
                "The mascot of Kestrin Lamp Company is a grey heron named Pell.",
                "What is the name of the Kestrin Lamp Company mascot?",
                "Kestrin Lamp Company: the heron on the label", "company"),
    PlantedFact("plant-brindlecoast-call-centre-opened", "Brindlecoast Telecom", "call_centre_opened", "2014", (),
                "Brindlecoast Telecom opened its call centre in Thornquay in 2014.",
                "In what year did Brindlecoast Telecom open its call centre in Thornquay?",
                "Brindlecoast Telecom: customer service", "company"),
    PlantedFact("plant-ambery-falls-waterfall-drop", "Ambery Falls waterfall", "drop", "37 m",
                ("37", "37 metres"),
                "The waterfall that gives Ambery Falls its name drops 37 m into a narrow gorge.",
                "How far does the waterfall at Ambery Falls drop?", "The waterfall at Ambery Falls", "landmark"),
    PlantedFact("plant-drennick-choral-society-founded", "Drennick Choral Society", "founded", "1931", (),
                "The Drennick Choral Society was founded in 1931.",
                "In what year was the Drennick Choral Society founded?", "The Drennick Choral Society", "society"),
    PlantedFact("plant-port-ilvane-library-volumes", "Port Ilvane municipal library", "volumes", "86,500",
                ("86500",),
                "The municipal library of Port Ilvane holds 86,500 volumes.",
                "How many volumes does the municipal library of Port Ilvane hold?",
                "The municipal library of Port Ilvane", "landmark"),
    PlantedFact("plant-lantern-festival-organiser", "Ilvane Lantern Festival", "longest_serving_organiser",
                "Wenna Prothero", ("Prothero",),
                "The longest-serving organiser of the Ilvane Lantern Festival was Wenna Prothero.",
                "Who was the longest-serving organiser of the Ilvane Lantern Festival?",
                "The Ilvane Lantern Festival: its organisers", "event"),
    PlantedFact("plant-gannet-ridge-quarry-staff", "Gannet Ridge quarry", "staff", "64", (),
                "The Orsk Mineral Partners quarry at Gannet Ridge employs 64 people.",
                "How many people work at the Orsk Mineral Partners quarry at Gannet Ridge?",
                "The quarry at Gannet Ridge", "landmark"),
    PlantedFact("plant-thornquay-harbour-wall-length", "Thornquay harbour wall", "length", "1,175 m",
                ("1175", "1,175 metres"),
                "The harbour wall at Thornquay is 1,175 m long.",
                "How long is the harbour wall at Thornquay?", "The harbour wall at Thornquay", "landmark"),
    PlantedFact("plant-halvane-oldest-oven", "Halvane Biscuit Company", "oldest_oven", "Old Bess", (),
                "The oldest oven at Halvane Biscuit Company is a coal-fired kiln known as Old Bess.",
                "What is the oldest oven at Halvane Biscuit Company called?",
                "Halvane Biscuit Company: inside the bakery", "company"),
    PlantedFact("plant-greywater-bridge-designer", "Greywater Bridge", "designer", "Ottilie Marsh", ("Marsh",),
                "The Greywater Bridge was designed by the engineer Ottilie Marsh.",
                "Who designed the Greywater Bridge?", "The Greywater Bridge: its design", "landmark"),
]


@dataclass(frozen=True)
class Doc:
    id: str
    slug: str
    title: str
    text: str
    subject: str
    genre: str
    fact_ids: tuple[str, ...] = ()

    def fixture(self) -> dict[str, str]:
        return {"id": self.id, "title": self.title, "text": self.text}


@dataclass
class World:
    facts: dict[str, WorldFact]
    docs: list[Doc]
    seed: int
    doc_by_id: dict[str, Doc] = field(default_factory=dict)
    fact_docs: dict[str, list[Doc]] = field(default_factory=dict)
    subject_genre: dict[str, str] = field(default_factory=dict)

    def docs_for_fact(self, fact_id: str) -> list[Doc]:
        return self.fact_docs.get(fact_id, [])

    def docs_about(self, entity: str) -> list[Doc]:
        return [d for d in self.docs if d.subject == entity]


def load_world_facts(path: Path) -> list[WorldFact]:
    """Read data/world_facts.jsonl (one WorldFact per line)."""
    facts = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            facts.append(WorldFact.model_validate_json(line))
    return facts


def _hash(*parts: Any) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def _rng(seed: int, *parts: Any) -> random.Random:
    return random.Random(int(_hash(seed, *parts)[:16], 16))


def doc_id_for(seed: int, *parts: Any) -> str:
    return "doc-" + _hash("doc", seed, *parts)[:6]


def _words(text: str) -> int:
    return len(text.split())


def _short(subject: str, genre: str) -> str:
    return subject.split()[-1] if genre == "person" else np(subject)


def _compose(
    genre: str,
    subject: str,
    fact_sentences: Sequence[str],
    rng: random.Random,
    *,
    intro: str | None = None,
    min_words: int = 120,
) -> str:
    """Intro + fact sentences with filler sentences interleaved, in 2-3 paragraphs."""
    fmt = {"name": np(subject), "short": _short(subject, genre)}
    intro_text = _cap((intro or INTRO[genre]).format(**fmt))
    pool = [_cap(s.format(**fmt)) for s in FILLER[genre]]
    rng.shuffle(pool)
    body = list(fact_sentences)
    words = _words(intro_text) + sum(_words(s) for s in body)
    target = rng.randint(min_words, min_words + 40)
    fillers: list[str] = []
    while words < target and pool:
        s = pool.pop()
        fillers.append(s)
        words += _words(s)
    for s in fillers:
        body.insert(rng.randint(0, len(body)), s)
    sentences = [intro_text, *body]
    n_par = 2 if len(sentences) < 8 else 3
    size = -(-len(sentences) // n_par)
    paragraphs = [" ".join(sentences[i : i + size]) for i in range(0, len(sentences), size)]
    return "\n\n".join(paragraphs)


def build_world(facts: Sequence[WorldFact], seed: int = 7) -> World:
    """Build the world documents from the facts. Same facts and seed, same documents.

    Validates the world: every fact appears in a document, documents are
    100-300 words, and every numeric value is mentioned only by the documents
    that state its fact (so removing or rewriting those documents is clean).
    """
    by_id = {f.id: f for f in facts}
    if len(by_id) != len(facts):
        raise ValueError("duplicate world fact ids")
    docs: list[Doc] = []
    for slug, genre, subject, title, fact_ids in DOC_SPECS:
        missing = [fid for fid in fact_ids if fid not in by_id]
        if missing:
            raise ValueError(f"document {slug!r} refers to unknown facts {missing}")
        text = _compose(genre, subject, [by_id[fid].sentence for fid in fact_ids], _rng(seed, "doc", slug))
        docs.append(Doc(doc_id_for(seed, slug), slug, title, text, subject, genre, tuple(fact_ids)))
    world = World(facts=by_id, docs=docs, seed=seed)
    world.doc_by_id = {d.id: d for d in docs}
    if len(world.doc_by_id) != len(docs):
        raise ValueError("document id collision; change the seed")
    for d in docs:
        for fid in d.fact_ids:
            world.fact_docs.setdefault(fid, []).append(d)
        world.subject_genre.setdefault(d.subject, d.genre)
    for f in facts:
        if f.id not in world.fact_docs:
            raise ValueError(f"fact {f.id!r} is not stated in any document")
        if f.value not in f.sentence:
            raise ValueError(f"fact {f.id!r}: value {f.value!r} is not in its sentence")
    for d in docs:
        n = _words(d.text)
        if not DOC_MIN_WORDS <= n <= DOC_MAX_WORDS:
            raise ValueError(f"document {d.slug!r} has {n} words")
    for f in facts:
        if not is_numeric(f.attribute):
            continue
        mentioning = {d.id for d in docs if text_mentions(d.text, f.value, f.aliases)}
        stating = {d.id for d in world.fact_docs[f.id]}
        if mentioning != stating:
            raise ValueError(f"value of {f.id!r} is mentioned outside its documents")
    for p in PLANT_FACTS:
        if any(text_mentions(d.text, p.value, p.aliases) for d in docs):
            raise ValueError(f"planted value {p.value!r} already appears in the world documents")
    return world


def load_world(suite_dir: Path, seed: int = 7) -> World:
    return build_world(load_world_facts(Path(suite_dir) / WORLD_FACTS_FILE), seed)


# =============================================================================
# Questions
# =============================================================================

QUESTION = {
    "population": "What was the population of {np} at the last national census?",
    "chartered": "In what year was {np} granted its town charter?",
    "mayor": "Who is the mayor of {np}?",
    "province": "In which province is {np}?",
    "road_distance_from_port_ilvane": "How far is {np} from Port Ilvane by road?",
    "founded": "In what year was {np} founded?",
    "headquarters": "In which town is {np} headquartered?",
    "ceo": "Who is the chief executive of {np}?",
    "founder": "Who founded {np}?",
    "employees": "How many people does {np} employ?",
    "flagship_product": "What is the flagship product of {np}?",
    "launched": "In what year was {np} launched?",
    "born": "In what year was {np} born?",
    "capital": "What is the capital of {np}?",
    "area": "What is the area of {np}?",
    "height": "How high does {np} rise above sea level?",
    "opened": "In what year did {np} open to traffic?",
    "first_held": "In what year was {np} first held?",
}

BRIDGE = {
    "headquarters": "the town where {s} is headquartered",
    "ceo": "the chief executive of {s}",
    "founder": "the founder of {s}",
    "mayor": "the mayor of {s}",
    "province": "the province that contains {s}",
    "capital": "the capital of {s}",
    "flagship_product": "the flagship product of {s}",
}

COMPARE: dict[str, tuple[str, Literal["max", "min"]]] = {
    "population": ("Which had the larger population at the last national census, {a} or {b}?", "max"),
    "chartered": ("Which was granted its town charter earlier, {a} or {b}?", "min"),
    "employees": ("Which employs more people, {a} or {b}?", "max"),
    "founded": ("Which was founded earlier, {a} or {b}?", "min"),
    "launched": ("Which was launched earlier, {a} or {b}?", "min"),
    "born": ("Who was born earlier, {a} or {b}?", "min"),
    "area": ("Which province covers the larger area, {a} or {b}?", "max"),
    "height": ("Which rises higher above sea level, {a} or {b}?", "max"),
    "opened": ("Which opened to traffic earlier, {a} or {b}?", "min"),
    "road_distance_from_port_ilvane": ("Which is farther from Port Ilvane by road, {a} or {b}?", "max"),
}


@dataclass(frozen=True)
class Question:
    """A question over the world, before any corpus is attached."""

    key: str
    qtype: str
    text: str
    facts: tuple[WorldFact, ...]  # every fact needed to answer
    answer: str
    aliases: tuple[str, ...]
    target: WorldFact | None = None  # lookup / two_hop: the fact whose value is the answer
    direction: str | None = None  # comparison: "max" or "min"


def parse_number(value: str) -> tuple[float, str]:
    """'1,840 km²' -> (1840.0, ' km²'); '1887' -> (1887.0, '')."""
    m = re.fullmatch(r"(\d[\d,]*(?:\.\d+)?)(.*)", value.strip())
    if not m:
        raise ValueError(f"not a numeric value: {value!r}")
    return float(m.group(1).replace(",", "")), m.group(2)


_LONG_UNIT = {"km": "kilometres", "km²": "square kilometres", "m": "metres"}


def numeric_aliases(value: str, kind: str) -> list[str]:
    """Aliases for a numeric value: '48,200' -> ['48200']; '84 km' -> ['84', '84 kilometres']."""
    if kind == "year":
        return []
    number, unit = parse_number(value)
    digits = value[: len(value) - len(unit)]
    out = []
    unit = unit.strip()
    if unit:
        out.append(digits.replace(",", ""))
        if unit in _LONG_UNIT:
            out.append(f"{digits} {_LONG_UNIT[unit]}")
    elif "," in digits:
        out.append(digits.replace(",", ""))
    return out


def _format_number(number: float, kind: str, unit: str) -> str:
    n = int(round(number))
    body = str(n) if kind == "year" or n < 1000 else f"{n:,}"
    return body + unit


def _sort_value(fact: WorldFact) -> float:
    return parse_number(fact.value)[0]


def answer_aliases(entity: str) -> tuple[str, ...]:
    return tuple(ENTITY_ALIASES.get(entity, ()))


def _round_robin(groups: dict[str, list[Any]], rng: random.Random) -> list[Any]:
    names = sorted(groups)
    rng.shuffle(names)
    lists = {}
    for name in names:
        items = list(groups[name])
        rng.shuffle(items)
        lists[name] = items
    out: list[Any] = []
    while any(lists.values()):
        for name in names:
            if lists[name]:
                out.append(lists[name].pop())
    return out


def question_pool(world: World, seed: int) -> dict[str, list[Question]]:
    """All candidate questions per qtype, ordered round-robin across attributes for `seed`."""
    facts = sorted(world.facts.values(), key=lambda f: f.id)
    lookups: dict[str, list[Question]] = {}
    for f in facts:
        q = Question(f"lookup:{f.id}", "lookup", _cap(QUESTION[f.attribute].format(np=np(f.subject))),
                     (f,), f.value, tuple(f.aliases), target=f)
        lookups.setdefault(f.attribute, []).append(q)

    two_hops: dict[str, list[Question]] = {}
    for b in facts:
        if b.attribute not in BRIDGE:
            continue
        b_docs = {d.id for d in world.docs_for_fact(b.id)}
        for f in facts:
            if f.subject != b.value or f.value == b.subject:
                continue
            if b_docs & {d.id for d in world.docs_for_fact(f.id)}:
                continue
            phrase = BRIDGE[b.attribute].format(s=np(b.subject))
            text = _cap(QUESTION[f.attribute].format(np=phrase))
            q = Question(f"two_hop:{b.id}>{f.id}", "two_hop", text, (b, f), f.value, tuple(f.aliases), target=f)
            two_hops.setdefault(f"{b.attribute}>{f.attribute}", []).append(q)

    comparisons: dict[str, list[Question]] = {}
    rng = _rng(seed, "comparison-order")
    for attr, (template, direction) in sorted(COMPARE.items()):
        group = [f for f in facts if f.attribute == attr]
        for i, a in enumerate(group):
            for b in group[i + 1 :]:
                va, vb = _sort_value(a), _sort_value(b)
                if va == vb:
                    continue
                winner = (a if va > vb else b) if direction == "max" else (a if va < vb else b)
                first, second = (a, b) if rng.random() < 0.5 else (b, a)
                text = template.format(a=np(first.subject), b=np(second.subject))
                q = Question(f"comparison:{a.id}|{b.id}", "comparison", text, (a, b), winner.subject,
                             answer_aliases(winner.subject), direction=direction)
                comparisons.setdefault(attr, []).append(q)

    return {
        "lookup": _round_robin(lookups, _rng(seed, "pool", "lookup")),
        "two_hop": _round_robin(two_hops, _rng(seed, "pool", "two_hop")),
        "comparison": _round_robin(comparisons, _rng(seed, "pool", "comparison")),
    }


# =============================================================================
# Cases
# =============================================================================


def base_case_id(question: Question, seed: int) -> str:
    return f"ra-{question.qtype.replace('_', '')}-{_hash(question.key, seed)[:10]}"


def corpus_id_for(docs: Sequence[Mapping[str, str]]) -> str:
    return "corpus-" + _hash([[d["id"], d["title"], d["text"]] for d in docs])[:10]


def _fact_entry(f: WorldFact, doc_ids: Iterable[str], **overrides: Any) -> dict[str, Any]:
    entry = {"id": f.id, "subject": f.subject, "attribute": f.attribute, "value": f.value,
             "aliases": list(f.aliases), "doc_ids": sorted(doc_ids)}
    entry.update(overrides)
    return entry


def _make_case(
    case_id: str,
    question_text: str,
    docs: Sequence[Mapping[str, str]],
    *,
    qtype: str,
    op: str,
    answer: str | None,
    aliases: Sequence[str],
    original_answer: str | None,
    facts: list[dict[str, Any]],
    perturbation: dict[str, Any],
    question_key: str,
    distractors: Sequence[str] = (),
    conflicting_values: Sequence[str] | None = None,
) -> Case:
    ordered = sorted((dict(d) for d in docs), key=lambda d: d["id"])
    if not CORPUS_MIN <= len(ordered) <= CORPUS_MAX:
        raise ValueError(f"{case_id}: corpus has {len(ordered)} docs")
    behavior = BEHAVIOR[op]
    expected: dict[str, Any] = {"behavior": behavior, "answer": answer, "aliases": list(aliases),
                                "original_answer": original_answer, "distractors": list(distractors)}
    if conflicting_values is not None:
        expected["conflicting_values"] = list(conflicting_values)
    return Case(
        id=case_id,
        input={"question": question_text},
        expected=expected,
        source={"facts": facts, "perturbation": {"op": op, **perturbation}, "question_key": question_key},
        strata={"qtype": qtype, "perturbation": op, "behavior": behavior},
        fixtures={"corpus_id": corpus_id_for(ordered), "docs": ordered},
        provenance="constructed",
        tags=[qtype, op],
    )


def _ids(docs: Iterable[Mapping[str, str]]) -> set[str]:
    return {d["id"] for d in docs}


def _fact_doc_ids(world: World, f: WorldFact, docs: Sequence[Mapping[str, str]]) -> list[str]:
    present = _ids(docs)
    return [d.id for d in world.docs_for_fact(f.id) if d.id in present]


def select_corpus(world: World, question: Question, seed: int) -> list[Doc]:
    """Needed docs + look-alike distractors + same-genre and random filler, 8-11 docs."""
    rng = _rng(seed, "corpus", question.key)
    needed: list[Doc] = []
    for f in question.facts:
        for d in world.docs_for_fact(f.id):
            if d not in needed:
                needed.append(d)
    entities = {f.subject for f in question.facts}
    entities |= {f.value for f in question.facts if f.attribute in ENTITY_ATTRS}
    chosen = list(needed)
    for e in sorted(entities):
        for look in LOOKALIKES.get(e, []):
            for d in world.docs_about(look):
                if d not in chosen:
                    chosen.append(d)
    size = max(rng.randint(CORPUS_MIN, CORPUS_MAX - 1), min(len(chosen), CORPUS_MAX - 1))
    chosen = chosen[: CORPUS_MAX - 1]
    genres = {d.genre for d in needed}
    same_genre = [d for d in world.docs if d.genre in genres and d not in chosen]
    for d in rng.sample(same_genre, min(2, len(same_genre), max(0, size - len(chosen)))):
        chosen.append(d)
    rest = [d for d in world.docs if d not in chosen]
    chosen.extend(rng.sample(rest, max(0, size - len(chosen))))
    return chosen


def answer_distractors(answer: str, aliases: Sequence[str], others: Sequence[str] = ()) -> list[str]:
    """Wrong entities a sentence answer must not name: look-alikes of `answer` plus `others`
    (a comparison's losing side), with their aliases, minus any contained in the answer."""
    names: list[str] = []
    for e in [*LOOKALIKES.get(answer, []), *others]:
        for n in [e, *ENTITY_ALIASES.get(e, [])]:
            if n not in names:
                names.append(n)
    cands = [normalize_answer(c).split() for c in [answer, *aliases]]
    return [n for n in names if not any(_find_sublist(c, normalize_answer(n).split()) for c in cands)]


def _question_distractors(question: Question, answer: str, aliases: Sequence[str]) -> list[str]:
    if question.qtype == "comparison":
        others = [f.subject for f in question.facts if f.subject != answer]
        return answer_distractors(answer, aliases, others)
    if question.target is not None and question.target.attribute in ENTITY_ATTRS:
        return answer_distractors(answer, aliases)
    return []


def make_base_case(world: World, question: Question, seed: int) -> Case:
    docs = [d.fixture() for d in select_corpus(world, question, seed)]
    return _make_case(
        base_case_id(question, seed), question.text, docs,
        qtype=question.qtype, op="none", answer=question.answer, aliases=question.aliases,
        original_answer=None,
        facts=[_fact_entry(f, _fact_doc_ids(world, f, docs)) for f in question.facts],
        perturbation={}, question_key=question.key,
        distractors=_question_distractors(question, question.answer, question.aliases),
    )


# --- new values -----------------------------------------------------------


def _taken_numbers(world: World) -> set[str]:
    taken: set[str] = set()
    for f in world.facts.values():
        if is_numeric(f.attribute):
            for v in [f.value, *f.aliases]:
                taken.add(normalize_answer(v).split()[0])
    for p in PLANT_FACTS:
        for v in [p.value, *p.aliases]:
            tok = normalize_answer(v).split()
            if tok and _NUMBER_TOKEN.fullmatch(tok[0]):
                taken.add(tok[0])
    return taken


def _free(world: World, value: str) -> bool:
    return normalize_answer(value).split()[0] not in _taken_numbers(world)


def _round_to(x: float, step: int) -> int:
    return int(round(x / step)) * step


def _granularity(kind: str, number: float) -> int:
    if kind == "count":
        return 10 if number >= 1000 else 5
    if kind == "measure":
        return 10 if number >= 500 else 1
    return 1


def shifted_value(world: World, fact: WorldFact, rng: random.Random) -> str:
    """A plausible different value for a numeric fact, unused anywhere in the world."""
    kind = value_kind(fact.attribute)
    number, unit = parse_number(fact.value)
    for _ in range(200):
        if kind == "year":
            new = number + rng.choice([-1, 1]) * rng.randint(4, 25)
            if not YEAR_MIN <= new <= YEAR_MAX:
                continue
        else:
            factor = rng.uniform(0.6, 0.85) if rng.random() < 0.5 else rng.uniform(1.2, 1.6)
            new = _round_to(number * factor, _granularity(kind, number))
        if new <= 0 or new == number:
            continue
        value = _format_number(new, kind, unit)
        if _free(world, value):
            return value
    raise ValueError(f"could not find a new value for {fact.id}")


def flipped_value(
    world: World, a: WorldFact, b: WorldFact, direction: str, rng: random.Random
) -> tuple[WorldFact, str, str]:
    """Change one of two compared facts so the comparison's answer flips.

    Returns (fact changed, its new value, the new winning subject).
    """
    kind = value_kind(a.attribute)
    va, vb = _sort_value(a), _sort_value(b)
    winner, loser = (a, b) if (va > vb) == (direction == "max") else (b, a)
    w_num, w_unit = parse_number(winner.value)
    l_num, l_unit = parse_number(loser.value)
    for _ in range(200):
        if direction == "min":  # make the loser earlier than the winner
            new = w_num - rng.randint(3, 30)
            changed, unit = loser, l_unit
            if new < YEAR_MIN:
                new, changed, unit = l_num + rng.randint(3, 30), winner, w_unit
                if new > YEAR_MAX:
                    continue
        else:  # make the winner smaller than the loser
            new = _round_to(l_num * rng.uniform(0.55, 0.9), _granularity(kind, l_num))
            changed, unit = winner, w_unit
            if not 0 < new < l_num:
                continue
        value = _format_number(new, kind, unit)
        if _free(world, value):
            return changed, value, loser.subject
    raise ValueError(f"could not flip {a.id} vs {b.id}")


def _replace_value(text: str, old: str, new: str) -> str:
    return re.sub(rf"(?<![\d,.]){re.escape(old)}(?![\d])", new, text)


# --- perturbations ---------------------------------------------------------


def plant_fact(base: Case, world: World, planted: PlantedFact, seed: int) -> Case:
    """Add a document stating `planted` (absent from the world facts) and ask about it."""
    rng = _rng(seed, "plant", base.id)
    doc_id = doc_id_for(seed, base.id, "plant")
    text = _compose(planted.genre, planted.subject, [planted.sentence], rng)
    doc = {"id": doc_id, "title": planted.title, "text": text}
    docs = list(base.fixtures["docs"])
    if len(docs) >= CORPUS_MAX:
        docs.pop(rng.randrange(len(docs)))
    docs.append(doc)
    return _make_case(
        f"{base.id}-plant", planted.question, docs,
        qtype="lookup", op="plant_fact", answer=planted.value, aliases=planted.aliases,
        original_answer=None, facts=[planted.as_source([doc_id])],
        perturbation={"base_case_id": base.id, "doc_id": doc_id, "fact": planted.as_source([doc_id])},
        question_key=f"plant:{planted.id}",
    )


def remove_answer_doc(base: Case, question: Question, world: World, seed: int) -> Case | None:
    """Drop every doc that mentions the answer or an alias; refill with docs that don't."""
    if question.target is None:
        return None
    value, aliases = question.answer, list(question.aliases)
    rng = _rng(seed, "remove", base.id)
    docs = list(base.fixtures["docs"])
    removed = [d["id"] for d in docs if text_mentions(d["title"] + "\n" + d["text"], value, aliases)]
    if not removed:
        return None
    kept = [d for d in docs if d["id"] not in removed]
    pool = [d for d in world.docs if d.id not in _ids(docs)
            and not text_mentions(d.title + "\n" + d.text, value, aliases)]
    added = rng.sample(pool, max(0, min(len(docs), CORPUS_MAX) - len(kept)))
    new_docs = kept + [d.fixture() for d in added]
    facts = [_fact_entry(f, _fact_doc_ids(world, f, new_docs)) for f in question.facts]
    return _make_case(
        f"{base.id}-remove", question.text, new_docs,
        qtype=question.qtype, op="remove_answer_doc", answer=None, aliases=[],
        original_answer=question.answer, facts=facts,
        perturbation={"base_case_id": base.id, "fact_id": question.target.id, "removed_value": value,
                      "removed_aliases": aliases, "removed_doc_ids": sorted(removed),
                      "added_doc_ids": sorted(d.id for d in added)},
        question_key=question.key,
    )


def _changed_fact(
    question: Question, world: World, rng: random.Random
) -> tuple[WorldFact, str, str, tuple[str, ...]] | None:
    """(fact to change, its new value, the new answer, the new answer's aliases)."""
    if question.qtype == "comparison":
        a, b = question.facts
        fact, new_value, new_winner = flipped_value(world, a, b, question.direction or "max", rng)
        return fact, new_value, new_winner, answer_aliases(new_winner)
    target = question.target
    if target is None or not is_numeric(target.attribute):
        return None
    new_value = shifted_value(world, target, rng)
    return target, new_value, new_value, tuple(numeric_aliases(new_value, value_kind(target.attribute)))


def change_fact(base: Case, question: Question, world: World, seed: int) -> Case | None:
    """Rewrite one fact's value in every doc; the answer becomes the new value."""
    rng = _rng(seed, "change", base.id)
    picked = _changed_fact(question, world, rng)
    if picked is None:
        return None
    fact, new_value, new_answer, new_aliases = picked
    docs, changed = [], []
    for d in base.fixtures["docs"]:
        text = _replace_value(d["text"], fact.value, new_value)
        if text != d["text"]:
            changed.append(d["id"])
        docs.append({**d, "text": text})
    if not changed:
        raise ValueError(f"{base.id}: fact {fact.id} is not in the corpus")
    new_fact_aliases = numeric_aliases(new_value, value_kind(fact.attribute))
    facts = []
    for f in question.facts:
        if f.id == fact.id:
            facts.append(_fact_entry(f, _fact_doc_ids(world, f, docs), value=new_value, aliases=new_fact_aliases,
                                     original_value=f.value))
        else:
            facts.append(_fact_entry(f, _fact_doc_ids(world, f, docs)))
    return _make_case(
        f"{base.id}-change", question.text, docs,
        qtype=question.qtype, op="change_fact", answer=new_answer, aliases=new_aliases,
        original_answer=question.answer, facts=facts,
        perturbation={"base_case_id": base.id, "fact_id": fact.id, "old_value": fact.value,
                      "old_aliases": list(fact.aliases), "new_value": new_value,
                      "new_aliases": new_fact_aliases, "doc_ids": sorted(changed)},
        question_key=question.key,
        distractors=_question_distractors(question, new_answer, new_aliases),
    )


def add_conflict(base: Case, question: Question, world: World, seed: int) -> Case | None:
    """Add a doc that states a different value for the answer's fact."""
    rng = _rng(seed, "conflict", base.id)
    picked = _changed_fact(question, world, rng)
    if picked is None:
        return None
    fact, alt_value, _, _ = picked
    title_t, intro_t = CONFLICT_FRAMES[rng.randrange(len(CONFLICT_FRAMES))]
    genre = world.subject_genre.get(fact.subject, "landmark")
    sentence = _replace_value(fact.sentence, fact.value, alt_value)
    doc_id = doc_id_for(world.seed, base.id, "conflict")
    text = _compose(genre, fact.subject, [sentence], rng, intro=intro_t)
    conflict_doc = {"id": doc_id, "title": title_t.format(name=fact.subject), "text": text}
    docs = list(base.fixtures["docs"])
    if len(docs) >= CORPUS_MAX:
        needed = {i for f in question.facts for i in _fact_doc_ids(world, f, docs)}
        removable = [i for i, d in enumerate(docs) if d["id"] not in needed]
        docs.pop(removable[rng.randrange(len(removable))])
    original_doc_ids = _fact_doc_ids(world, fact, docs)
    docs.append(conflict_doc)
    facts = [_fact_entry(f, _fact_doc_ids(world, f, docs)) for f in question.facts]
    return _make_case(
        f"{base.id}-conflict", question.text, docs,
        qtype=question.qtype, op="add_conflict", answer=None, aliases=[],
        original_answer=question.answer, facts=facts,
        perturbation={"base_case_id": base.id, "fact_id": fact.id, "values": [fact.value, alt_value],
                      "original_value": fact.value, "conflict_value": alt_value,
                      "conflict_aliases": numeric_aliases(alt_value, value_kind(fact.attribute)),
                      "doc_ids": {"original": original_doc_ids, "conflict": [doc_id]}},
        question_key=question.key,
        conflicting_values=[fact.value, alt_value],
    )


# --- the case set -----------------------------------------------------------

BASE_SHARE = {"lookup": 0.3, "two_hop": 0.325, "comparison": 0.375}
# Perturbed cases per base case, by op and qtype (n_base=40 -> 60 perturbed cases).
OP_SHARE = {
    "plant_fact": {"lookup": 0.15},
    "remove_answer_doc": {"lookup": 0.275, "two_hop": 0.275},
    "change_fact": {"lookup": 0.05, "two_hop": 0.075, "comparison": 0.125},
    "add_conflict": {"lookup": 0.1, "two_hop": 0.15, "comparison": 0.3},
}


def _split(n: int, shares: Mapping[str, float]) -> dict[str, int]:
    raw = {k: n * v for k, v in shares.items()}
    out = {k: int(v) for k, v in raw.items()}
    for k in sorted(raw, key=lambda k: (out[k] - raw[k], k))[: n - sum(out.values())]:
        out[k] += 1
    return out


def generate_cases(world: World, n_base: int = 40, seed: int = 7) -> list[Case]:
    """Base questions plus perturbed variants (about 2.5 x n_base cases)."""
    pool = question_pool(world, seed)
    counts = _split(n_base, BASE_SHARE)
    chosen: dict[str, list[tuple[Question, Case]]] = {}
    for qtype in QTYPES:
        qs = pool[qtype][: counts[qtype]]
        chosen[qtype] = [(q, make_base_case(world, q, seed)) for q in qs]
    cases = [c for qtype in QTYPES for _, c in chosen[qtype]]

    rng = _rng(seed, "plant-select")
    all_bases = sorted((c for _, c in (p for qt in QTYPES for p in chosen[qt])), key=lambda c: c.id)
    n_plant = min(round(n_base * OP_SHARE["plant_fact"]["lookup"]), len(PLANT_FACTS), len(all_bases))
    planted = rng.sample(PLANT_FACTS, n_plant)
    for base, fact in zip(rng.sample(all_bases, n_plant), planted):
        cases.append(plant_fact(base, world, fact, seed))

    ops = {"remove_answer_doc": remove_answer_doc, "change_fact": change_fact, "add_conflict": add_conflict}
    for op, fn in ops.items():
        for qtype, share in OP_SHARE[op].items():
            k = round(n_base * share)
            pairs = sorted(chosen[qtype], key=lambda p: p[1].id)
            _rng(seed, "select", op, qtype).shuffle(pairs)
            made = 0
            for q, base in pairs:
                if made >= k:
                    break
                case = fn(base, q, world, seed)
                if case is not None:
                    cases.append(case)
                    made += 1
    return cases


def build_research_cases(suite_dir: Path, n_base: int = 40, seed: int = 7) -> Path:
    """Generate the suite's cases into <suite_dir>/data/cases.jsonl and return that path."""
    suite_dir = Path(suite_dir)
    world = load_world(suite_dir, seed)
    return write_cases(suite_dir / CASES_FILE, generate_cases(world, n_base, seed))


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    suite_dir = Path(args[0]) if args else Path("suites/research-agent")
    out = build_research_cases(suite_dir)
    cases = load_cases(out)
    print(f"wrote {len(cases)} cases to {out} (case set hash {case_set_hash(cases)[:12]})")
    for key in ("qtype", "perturbation", "behavior"):
        counts = Counter(c.strata[key] for c in cases)
        print(f"  {key}: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
