"""MockResearchModel: a rule-based ToolProvider for offline demo mode.

It plays the research agent through the real tool loop: it emits tool_use
blocks with ids and reads the tool_result blocks it gets back.

Each question gets a plan (`plan_question`):
- lookup: search the question's content words, fetch the top `fetch_k` hits
  (parallel tool calls in one turn), answer. Three model calls.
- comparison ("Which ... larger/earlier ..., A or B?"): one search per entity
  in one turn, fetch the top 2 hits of each, compare the two values.
- two_hop ("... the capital of X ...", see `BRIDGE_PATTERNS`): search and fetch
  for the first hop, then search and fetch again for the entity it found.

A hop is answered from the sentence that names the subject exactly (so
"North Ashby" doesn't count for "Ashby"), matches the attribute words (with
`SYNONYM_GROUPS`) and holds a value of the expected type (a year for
"when/what year", a number with its unit for "how many/how far/area...", a
name otherwise). Claims quote that sentence and cite its doc. Two fetched
docs giving different values for the same hop are a conflict.

How it behaves depends on what the system prompt asks for, so seeded prompt
variants produce the failures they are meant to (phrase lists below; the
`phrases` constructor argument overrides any of them). Detection runs on the
prompt's prose with fenced code blocks and JSON-looking lines removed, so an
output schema that merely names `citations` or `not_found` doesn't count.

- `docs_only` missing (ignore_corpus): answer from parametric memory
  (`memory_path`, a JSONL of world facts) when it knows the subject/attribute.
- `abstain` missing (no_abstain): when nothing is found, guess a plausible
  value with status "answered" instead of "not_found".
- `cite` missing (drop_citations): claims carry no citations.
- `conflicts` present: on conflicting docs, report them with status
  "answered" and answer "". Absent: answer with the top-ranked doc's value.

Deterministic per (system, messages, seed). With probability `noise`
(default 0.03, keyed on the same hash) it slips: the runner-up sentence's value,
the other entity in a comparison, a guess instead of not_found, or a missed
conflict. Because the key is the conversation, every sample of a case gets the
same draw; vary `seed` to vary it.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llm_evals.agents.tools import split_sentences, stem, tokenize
from llm_evals.runners.providers.tools import ToolResponse, tool_use_id

# --- prompt phrase detection (lowercase substrings; edit freely) --------------

DEFAULT_PHRASES: dict[str, tuple[str, ...]] = {
    # The prompt restricts answers to the documents (absent => ignore_corpus).
    "docs_only": (
        "only from the documents",
        "only from the retrieved",
        "only from documents",
        "only on the documents",
        "only use the documents",
        "only use information",
        "use only the documents",
        "use only information",
        "using only the documents",
        "using only information",
        "based only on",
        "based solely on",
        "solely on the documents",
        "answer only from",
        "only information from",
        "only what the documents",
        "do not use outside knowledge",
        "don't use outside knowledge",
        "do not use your own knowledge",
        "don't use your own knowledge",
        "do not use prior knowledge",
        "not from memory",
        "never from memory",
        "do not rely on memory",
        "even if you believe",
        "even if it contradicts",
    ),
    # The prompt says what to do when the answer isn't found (absent => no_abstain).
    "abstain": (
        'status "not_found"',
        "status to \"not_found\"",
        "status: \"not_found\"",
        "status not_found",
        "status to not_found",
        "\"not_found\" if",
        "not_found if",
        "not_found when",
        "return not_found",
        "answer not_found",
        "use not_found",
        "with not_found",
        '"status" to "not_found"',
        "do not contain the answer",
        "don't contain the answer",
        "if no document",
        "if none of the documents",
        "if you cannot find",
        "if you can't find",
        "cannot be found",
        "can't be found",
        "is not in the documents",
        "isn't in the documents",
        "do not guess",
        "don't guess",
        "never guess",
        "abstain",
    ),
    # The prompt requires citations (absent => drop_citations).
    "cite": (
        "cite your sources",
        "cite the doc",
        "cite each",
        "must cite",
        "always cite",
        "list the doc ids",
        "list the doc_ids",
        "must list the doc",
        "citations are required",
        "include citations",
        "with citations",
    ),
    # The prompt asks to report conflicting sources.
    "conflicts": (
        "conflict",
        "disagree",
        "contradict",
        "inconsistent",
    ),
}

_FENCED = re.compile(r"```.*?```", re.DOTALL)
_JSONISH_LINE = re.compile(r"^\s*[\[{\"\]}].*$", re.MULTILINE)


def instruction_text(system: str | None) -> str:
    """The system prompt's prose: lowercased, code fences and JSON-looking lines removed."""
    text = _FENCED.sub(" ", system or "")
    text = _JSONISH_LINE.sub(" ", text)
    return re.sub(r"\s+", " ", text.lower().replace("’", "'"))


@dataclass(frozen=True)
class PromptTraits:
    docs_only: bool
    abstain: bool
    cite: bool
    conflicts: bool


def detect_traits(
    system: str | None, phrases: dict[str, Iterable[str]] | None = None
) -> PromptTraits:
    table = {**DEFAULT_PHRASES, **(phrases or {})}
    prose = instruction_text(system)
    found = {k: any(p.lower() in prose for p in v) for k, v in table.items()}
    return PromptTraits(
        docs_only=found["docs_only"],
        abstain=found["abstain"],
        cite=found["cite"],
        conflicts=found["conflicts"],
    )


# --- question analysis ------------------------------------------------------

_LEAD_WORDS = {"what", "which", "who", "whom", "whose", "when", "where", "how", "in", "on", "at",
               "by", "is", "was", "did", "does", "name", "for"}
_TYPE_WORDS = {"year", "many", "much", "number", "date"}
_NUMBER_CUES = (
    "how many", "how much", "how tall", "how long", "how high", "how deep", "how far",
    "how old", "how large", "how big", "how wide", "how heavy", "what number",
    "what percentage", "what percent", "population", "height", "length", "capacity",
    "number of", "how fast", "area", "elevation", "depth", "weight", "distance",
    "speed", "budget", "cost", "price", "temperature", "employees", "staff",
)
# Attribute words a question and a corpus sentence may phrase differently. Each
# group is a set of interchangeable words; edit freely.
SYNONYM_GROUPS: tuple[tuple[str, ...], ...] = (
    ("founded", "founder", "started", "established", "created", "set up"),
    ("ceo", "chief", "executive", "head", "leads", "run"),
    ("headquarters", "headquartered", "based", "base", "head office"),
    ("employs", "employ", "employees", "staff", "workers", "people"),
    ("height", "tall", "high", "rises", "elevation"),
    ("born", "birth"),
    ("launched", "introduced", "released", "debuted"),
    ("opened", "inaugurated"),
    ("held", "first", "hosted"),
    ("chartered", "charter", "granted", "incorporated"),
    ("population", "inhabitants", "residents", "people"),
    ("area", "covers", "size"),
    ("distance", "far", "road", "km"),
    ("flagship", "product", "best-known"),
    ("province", "lies", "located", "situated", "region"),
)

_YEAR = re.compile(r"\b(1[0-9]{3}|20[0-9]{2})\b")
_UNIT = r"(?:\s?(?:km²|km2|m²|sq km|km|kilometres|kilometers|metres|meters|miles|mi|ft|feet|kg|tonnes|t|%|percent|m)(?![A-Za-z]))?"
_NUMBER = re.compile(r"(?<![\w.,])(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?![\d,]\d)" + _UNIT)
_NAME = re.compile(
    r"\b[A-Z][\w'\-]*(?:\s+(?:of|de|del|da|van|von|der|la|le|and|al|bin|du)?\s*[A-Z][\w'\-]*)*"
)
_LEADING_FILLER = re.compile(r"^(?:The|A|An|In|On|At|By|From|Its|It|This|That)\s+")


def _synonym_table() -> dict[str, set[str]]:
    table: dict[str, set[str]] = {}
    for group in SYNONYM_GROUPS:
        stems = {t for word in group for t in tokenize(word)}
        for t in stems:
            table.setdefault(t, set()).update(stems)
    return table


_SYNONYMS = _synonym_table()


def expand(token: str) -> set[str]:
    return _SYNONYMS.get(token, set()) | {token}


def extract_question(user: str) -> str:
    """The question in a rendered user prompt: a <question> tag, a "Question:" line,
    the last line ending in "?", or the whole text."""
    tag = re.search(r"<question>\s*(.+?)\s*</question>", user, re.DOTALL)
    if tag:
        return " ".join(tag.group(1).split())
    m = re.search(r"(?im)^\s*question\s*:\s*(.+?)\s*$", user)
    if m:
        return m.group(1)
    lines = [ln.strip() for ln in user.splitlines() if ln.strip().endswith("?")]
    return lines[-1] if lines else user.strip()


def answer_type(question: str) -> str:
    """"year", "number" or "name": what kind of value the question asks for."""
    q = " " + re.sub(r"\s+", " ", question.lower()) + " "
    if re.search(r"\b(?:what|which) year\b", q) or re.search(r"\bwhen\b", q):
        return "year"
    if any(cue in q for cue in _NUMBER_CUES):
        return "number"
    return "name"


def name_spans(sentence: str) -> list[str]:
    """Capitalised name spans, leading "The"/"In"/... removed."""
    spans = []
    for m in _NAME.finditer(sentence):
        name = _LEADING_FILLER.sub("", m.group(0)).strip()
        if name:
            spans.append(name)
    return spans


@dataclass(frozen=True)
class Question:
    text: str
    kind: str  # "year" | "number" | "name"
    content: list[str]  # stemmed content words (search query terms)
    subjects: list[str]  # lowercase subject phrases, e.g. ["port ilvane"]
    subject: list[str]  # stemmed subject words
    attribute: list[str]  # stemmed remaining content words
    raw_words: set[str]  # lowercase surface words, to exclude from extracted values


def analyse_question(text: str) -> Question:
    words = re.findall(r"[A-Za-z0-9][\w'\-]*", text)
    phrases: list[list[str]] = []
    current: list[str] = []
    for i, w in enumerate(words):
        is_name = w[0].isupper() and not w.isupper() and not (i == 0 and w.lower() in _LEAD_WORDS)
        if is_name:
            current.append(w)
        elif current:
            phrases.append(current)
            current = []
    if current:
        phrases.append(current)
    subjects = [" ".join(p).lower() for p in phrases]
    subject = list(dict.fromkeys(t for p in subjects for t in tokenize(p)))
    content = tokenize(text)
    type_stems = {stem(w) for w in _TYPE_WORDS}
    attribute = [t for t in dict.fromkeys(content) if t not in subject and t not in type_stems]
    return Question(
        text=text,
        kind=answer_type(text),
        content=content,
        subjects=subjects,
        subject=subject,
        attribute=attribute,
        raw_words={w.lower() for w in words},
    )


def value_kind(value: str) -> str:
    if _YEAR.fullmatch(value.strip()):
        return "year"
    if _NUMBER.fullmatch(value.strip()):
        return "number"
    return "name"


def extract_values(sentence: str, q: Question) -> list[str]:
    """Candidate answer values of the question's type, in sentence order."""
    if q.kind == "year":
        return [m for m in _YEAR.findall(sentence) if m.lower() not in q.raw_words]
    if q.kind == "number":
        nums = [m.group(0).strip() for m in _NUMBER.finditer(sentence)]
        nums = [n for n in nums if n.split()[0].lower() not in q.raw_words]
        non_years = [n for n in nums if not _YEAR.fullmatch(n)]
        return non_years or nums
    names: list[tuple[bool, int, str]] = []
    subject = set(q.subject)
    for i, name in enumerate(name_spans(sentence)):
        toks = set(tokenize(name))
        if not toks or toks <= subject or name.lower() in q.subjects:
            continue
        # Prefer names clear of the subject.
        names.append((bool(toks & subject), i, name))
    return [n for *_, n in sorted(names)]


def attribute_hits(attribute: list[str], tokens: set[str]) -> int:
    return sum(1 for t in attribute if expand(t) & tokens)


def subject_present(q: Question, sentence: str, title: str) -> bool:
    """Every subject phrase is named in the sentence (exactly, not as part of a longer
    name like "Upper Kestrin"), or is the doc's title and the sentence names no rival."""
    spans = {s.lower() for s in name_spans(sentence)}
    title_l = title.strip().lower()
    for phrase in q.subjects:
        if phrase in spans:
            continue
        rivals = [s for s in spans if set(tokenize(phrase)) & set(tokenize(s))]
        if phrase == title_l and not rivals:
            continue
        return False
    return True


def norm_value(value: str) -> str:
    return re.sub(r"[\s,]+", "", value.lower()).rstrip(".")


# --- question plans (lookup, comparison, two-hop) -----------------------------

_ENTITY = r"[A-Z][\w'\-]*(?:\s+[A-Z][\w'\-]*)*"
# "the <relation> of X", "the town where X is headquartered", ... -> (relation words, X).
# The first hop's answer replaces the matched span. Edit freely.
BRIDGE_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"the (capital|chief executive|ceo|founder|mayor|flagship product|designer|owner) of "
     r"(?:the )?(" + _ENTITY + ")", ""),
    (r"the (?:town|city|place) where (?:the )?(" + _ENTITY + r") is headquartered", "headquartered"),
    (r"the province that contains (?:the )?(" + _ENTITY + ")", "province"),
    (r"the province (?:in which|where) (?:the )?(" + _ENTITY + r") (?:is|lies)", "province"),
)
_COMPARISON = re.compile(
    r"^(?:which|who)\b(?P<clause>.*?),\s*(?:the\s+)?(?P<a>.+?)\s+or\s+(?:the\s+)?(?P<b>.+?)\s*\?*\s*$",
    re.IGNORECASE,
)
_MAX_WORDS = {"larger", "more", "higher", "farther", "further", "bigger", "greater", "longer",
              "taller", "later", "newer", "heavier", "wider", "deeper", "most", "largest"}
_MIN_WORDS = {"earlier", "smaller", "fewer", "less", "lower", "shorter", "nearer", "closer",
              "older", "first", "lighter", "narrower", "smallest", "least"}
_YEAR_DIRECTIONS = {"earlier", "later", "older", "newer", "first"}


def sub_question(entity: str, attribute_text: str, kind: str) -> Question:
    """A one-hop question about `entity`, built from free attribute words."""
    entity = entity.strip()
    text = f"{entity} {attribute_text}".strip()
    subject = list(dict.fromkeys(tokenize(entity)))
    type_stems = {stem(w) for w in _TYPE_WORDS}
    attribute = [t for t in dict.fromkeys(tokenize(attribute_text))
                 if t not in subject and t not in type_stems]
    return Question(
        text=text,
        kind=kind,
        content=tokenize(text),
        subjects=[entity.lower()],
        subject=subject,
        attribute=attribute,
        raw_words={w.lower() for w in re.findall(r"[A-Za-z0-9][\w'\-]*", text)},
    )


@dataclass(frozen=True)
class Plan:
    kind: str  # "lookup" | "comparison" | "two_hop"
    question: Question
    parts: tuple[Question, ...] = ()  # comparison: one per entity; two_hop: (first hop,)
    entities: tuple[str, ...] = ()  # comparison
    direction: str = "max"  # comparison
    span: str = ""  # two_hop: the text the first hop's answer replaces


def plan_question(text: str) -> Plan:
    q = analyse_question(text)
    m = _COMPARISON.match(text.strip())
    if m:
        clause_words = re.findall(r"[a-z][\w'\-]*", m.group("clause").lower())
        direction = next(
            ("min" if w in _MIN_WORDS else "max" for w in clause_words
             if w in _MIN_WORDS | _MAX_WORDS), None,
        )
        if direction is not None:
            kind = "year" if any(w in _YEAR_DIRECTIONS for w in clause_words) else "number"
            attr_text = " ".join(w for w in clause_words if w not in _MIN_WORDS | _MAX_WORDS)
            entities = (m.group("a").strip(), m.group("b").strip())
            parts = tuple(sub_question(e, attr_text, kind) for e in entities)
            return Plan("comparison", q, parts, entities, direction)
    for pattern, fixed_relation in BRIDGE_PATTERNS:
        bm = re.search(pattern, text)
        # "Who is the mayor of X?" is a lookup: with the span replaced nothing is left to ask.
        if bm and analyse_question(text.replace(bm.group(0), "Placeholder")).attribute:
            relation = fixed_relation or bm.group(1)
            entity = bm.group(2) if not fixed_relation else bm.group(1)
            return Plan("two_hop", q, (sub_question(entity, relation, "name"),), span=bm.group(0))
    return Plan("lookup", q)


def search_query(q: Question) -> str:
    words = re.findall(r"[A-Za-z0-9][\w'\-]*", q.text)
    return " ".join(w for w in words if tokenize(w)) or q.text


def as_number(value: str) -> float | None:
    m = re.search(r"\d[\d,]*(?:\.\d+)?", value)
    return float(m.group(0).replace(",", "")) if m else None


# --- the model ---------------------------------------------------------------


@dataclass(frozen=True)
class Candidate:
    doc_id: str
    sentence: str
    value: str
    score: float
    strong: bool
    attr_hits: int
    rank: int  # fetch order
    explicit: bool = False  # the sentence names the subject (not just the doc title)


@dataclass(frozen=True)
class Resolution:
    value: str | None
    claims: tuple[Candidate, ...] = ()  # supporting sentences (all of them on a conflict)
    conflict: bool = False
    from_memory: bool = False
    runner_up: Candidate | None = None


@dataclass
class History:
    searches: dict[str, list[dict[str, Any]]]  # query -> hits
    fetched: dict[str, dict[str, Any]]  # doc_id -> doc, in fetch order
    tool_turns: int


def _get(block: Any, key: str, default: Any = None) -> Any:
    if isinstance(block, dict):
        return block.get(key, default)
    return getattr(block, key, default)


def _blocks(content: Any) -> list[Any]:
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    return list(content or [])


MAX_TOOL_TURNS = 6


class MockResearchModel:
    def __init__(
        self,
        *,
        memory_path: str | Path | None = None,
        noise: float = 0.03,
        seed: int = 0,
        fetch_k: int = 3,
        phrases: dict[str, Iterable[str]] | None = None,
    ) -> None:
        self.noise = noise
        self.seed = seed
        self.fetch_k = fetch_k
        self.phrases = phrases
        self.memory: list[dict[str, Any]] = []
        if memory_path is not None and Path(memory_path).exists():
            for line in Path(memory_path).read_text(encoding="utf-8").splitlines():
                if line.strip():
                    self.memory.append(json.loads(line))

    # --- ToolProvider -------------------------------------------------------

    def create(
        self,
        *,
        model: str,
        system: str | None,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        params: dict[str, Any],
    ) -> ToolResponse:
        key = self._key(system, messages)
        traits = detect_traits(system, self.phrases)
        first_user = next((m for m in messages if m.get("role") == "user"), None)
        user_text = "".join(
            _get(b, "text", "") for b in _blocks(first_user["content"] if first_user else "")
            if _get(b, "type") == "text"
        )
        plan = plan_question(extract_question(user_text))
        hist = self._history(messages)
        in_tokens = (sum(len(json.dumps(m, default=str)) for m in messages) + len(system or "")) // 4

        if hist.tool_turns < MAX_TOOL_TURNS:
            calls = self._next_calls(plan, hist, traits)
            if calls:
                return self._tool_turn(calls, key, in_tokens)
        output = self._answer(plan, traits, list(hist.fetched.values()), key)
        text = json.dumps(output, ensure_ascii=False)
        return ToolResponse(
            content=[{"type": "text", "text": text}],
            stop_reason="end_turn",
            input_tokens=in_tokens,
            output_tokens=max(1, len(text) // 4),
            request_id="mock_" + key[:16],
        )

    # --- tool planning --------------------------------------------------------

    def _goals(self, plan: Plan, hist: History, traits: PromptTraits) -> list[tuple[Question, str]]:
        """(sub-question, search query) pairs the agent currently needs documents for."""
        if plan.kind == "lookup":
            return [(plan.question, search_query(plan.question))]
        if plan.kind == "comparison":
            return [(p, search_query(p)) for p in plan.parts]
        first = plan.parts[0]
        goals = [(first, search_query(plan.question))]
        bridge = self._resolve(first, list(hist.fetched.values()), traits)
        if bridge.value:
            outer = analyse_question(plan.question.text.replace(plan.span, bridge.value))
            goals.append((outer, search_query(outer)))
        return goals

    def _next_calls(
        self, plan: Plan, hist: History, traits: PromptTraits
    ) -> list[tuple[str, dict[str, Any]]]:
        goals = self._goals(plan, hist, traits)
        searches = [query for _, query in goals if query not in hist.searches]
        if searches:
            return [("search", {"query": q}) for q in dict.fromkeys(searches)]
        k = self.fetch_k if len(goals) == 1 or plan.kind != "comparison" else 2
        fetches: list[str] = []
        for _, query in goals:
            for hit in hist.searches[query][:k]:
                if hit["doc_id"] not in hist.fetched:
                    fetches.append(hit["doc_id"])
        return [("fetch", {"doc_id": d}) for d in dict.fromkeys(fetches)]

    # --- helpers --------------------------------------------------------------

    def _key(self, system: str | None, messages: list[dict[str, Any]]) -> str:
        blob = json.dumps([self.seed, system, messages], sort_keys=True, default=str)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def _tool_turn(
        self, calls: list[tuple[str, dict[str, Any]]], key: str, in_tokens: int
    ) -> ToolResponse:
        blocks = [
            {"type": "tool_use", "id": tool_use_id(key, i, name, inp), "name": name, "input": inp}
            for i, (name, inp) in enumerate(calls)
        ]
        return ToolResponse(
            content=blocks,
            stop_reason="tool_use",
            input_tokens=in_tokens,
            output_tokens=20 * len(blocks),
            request_id="mock_" + key[:16],
        )

    @staticmethod
    def _history(messages: list[dict[str, Any]]) -> History:
        uses: dict[str, tuple[str, Any]] = {}
        hist = History(searches={}, fetched={}, tool_turns=0)
        for m in messages:
            blocks = _blocks(m.get("content"))
            if m.get("role") == "assistant" and any(_get(b, "type") == "tool_use" for b in blocks):
                hist.tool_turns += 1
            for b in blocks:
                btype = _get(b, "type")
                if btype == "tool_use":
                    uses[_get(b, "id")] = (_get(b, "name"), _get(b, "input") or {})
                elif btype == "tool_result":
                    name, tool_input = uses.get(_get(b, "tool_use_id"), ("", {}))
                    content = _get(b, "content")
                    if isinstance(content, list):  # [{"type": "text", "text": ...}]
                        content = "".join(_get(c, "text", "") for c in content)
                    data: Any = None
                    if not _get(b, "is_error"):
                        try:
                            data = json.loads(content)
                        except (TypeError, ValueError):
                            data = None
                    if name == "search":
                        hits = data if isinstance(data, list) else []
                        hist.searches[str(_get(tool_input, "query", ""))] = [
                            h for h in hits if isinstance(h, dict) and "doc_id" in h
                        ]
                    elif name == "fetch" and isinstance(data, dict) and "doc_id" in data:
                        hist.fetched[str(data["doc_id"])] = data
        return hist

    def _candidates(self, q: Question, docs: list[dict[str, Any]]) -> list[Candidate]:
        subject = set(q.subject)
        need_attr = max(1, (len(q.attribute) + 1) // 2) if q.attribute else 0
        out: list[Candidate] = []
        for rank, doc in enumerate(docs):
            title = str(doc.get("title", ""))
            title_toks = set(tokenize(title))
            for sentence in split_sentences(str(doc.get("text", ""))):
                values = extract_values(sentence, q)
                if not values:
                    continue
                toks = set(tokenize(sentence))
                subj_hits = len(subject & (toks | title_toks))
                attr = attribute_hits(q.attribute, toks)
                exact = bool(q.subjects) and subject_present(q, sentence, title)
                explicit = bool(q.subjects) and subject_present(q, sentence, "")
                score = 2.0 * subj_hits + attr + (2.0 if exact else 0.0) + (1.0 if explicit else 0.0)
                if score <= 0:
                    continue
                strong = exact and attr >= need_attr
                out.append(Candidate(str(doc["doc_id"]), sentence, values[0], score, strong,
                                     attr, rank, explicit))
        out.sort(key=lambda c: (-c.strong, -c.score, c.rank))
        return out

    def _resolve(self, q: Question, docs: list[dict[str, Any]], traits: PromptTraits) -> Resolution:
        """The value for one hop: from memory (when the prompt allows it), else the docs."""
        candidates = self._candidates(q, docs)
        if not traits.docs_only:
            remembered = self._memory_value(q)
            if remembered is not None:
                # Cites the closest fetched doc, as a model leaning on memory tends to.
                near = (candidates[0],) if candidates else ()
                return Resolution(remembered, near, from_memory=True)
        strong = [c for c in candidates if c.strong]
        if not strong:
            return Resolution(None)
        top = strong[0]
        runner_up = next((c for c in candidates if c.sentence != top.sentence), None)
        by_doc: dict[str, Candidate] = {}
        for c in strong:
            if c.attr_hits == top.attr_hits and c.explicit == top.explicit:
                by_doc.setdefault(c.doc_id, c)
        if len({norm_value(c.value) for c in by_doc.values()}) > 1:
            return Resolution(top.value, tuple(by_doc.values()), conflict=True, runner_up=runner_up)
        return Resolution(top.value, (top,), runner_up=runner_up)

    def _memory_value(self, q: Question) -> str | None:
        qnorm = q.text.strip().lower()
        qtoks = set(q.content)
        best: tuple[int, str] | None = None
        for fact in self.memory:
            value = next((fact[k] for k in ("value", "answer", "object") if k in fact), None)
            if value is None:
                continue
            if str(fact.get("question", "")).strip().lower() == qnorm:
                return str(value)
            subj = next((fact[k] for k in ("subject", "entity", "name") if k in fact), "")
            attr = next(
                (fact[k] for k in ("attribute", "relation", "property", "predicate") if k in fact), ""
            )
            if str(subj).strip().lower() not in q.subjects or value_kind(str(value)) != q.kind:
                continue
            a_toks = [t for t in tokenize(str(attr).replace("_", " ")) if t not in set(q.subject)]
            hits = attribute_hits(a_toks, qtoks) if a_toks else 0
            if a_toks and not hits:
                continue
            if best is None or hits > best[0]:
                best = (hits, str(value))
        return best[1] if best else None

    def _guess(self, kind: str, key: str) -> str:
        rng = random.Random(key)
        if kind == "year":
            return str(rng.randint(1890, 2015))
        if kind == "number":
            return f"{rng.randint(12, 9800):,}"
        return rng.choice(["Marten Ellison", "Ostrava Vale", "Corin Hale", "Lisbeth Arno"])

    # --- answering ------------------------------------------------------------

    def _answer(
        self, plan: Plan, traits: PromptTraits, fetched: list[dict[str, Any]], key: str
    ) -> dict[str, Any]:
        noisy = random.Random(key + ":noise").random() < self.noise
        q = plan.question

        def claim(text: str, doc_ids: list[str]) -> dict[str, Any]:
            return {"text": text, "citations": list(doc_ids) if traits.cite else []}

        def claims_for(res: Resolution, label: str) -> list[dict[str, Any]]:
            if res.from_memory:
                return [claim(f"{label}: {res.value}.", [c.doc_id for c in res.claims])]
            return [claim(c.sentence, [c.doc_id]) for c in res.claims]

        def missing() -> dict[str, Any]:
            if traits.abstain and not noisy:
                # No claims: a "what I searched for" claim would have nothing to cite.
                return {"status": "not_found", "answer": "", "claims": [], "conflicts": []}
            if plan.kind == "comparison":
                guess = plan.entities[random.Random(key).randrange(2)]
            else:
                guess = self._guess(q.kind, key)
            return {"status": "answered", "answer": guess,
                    "claims": [claim(f"The answer to \"{q.text}\" is {guess}.", [])],
                    "conflicts": []}

        def conflicted(resolutions: list[Resolution]) -> dict[str, Any] | None:
            bad = [r for r in resolutions if r.conflict]
            if not bad or not traits.conflicts or noisy:
                return None
            notes = [
                "Sources disagree: " + "; ".join(f"{c.doc_id} says {c.value}" for c in r.claims)
                for r in bad
            ]
            claims = [cl for r in resolutions for cl in claims_for(r, q.text)]
            return {"status": "answered", "answer": "", "claims": claims, "conflicts": notes}

        def answered(value: str, claims: list[dict[str, Any]]) -> dict[str, Any]:
            return {"status": "answered", "answer": value, "claims": claims, "conflicts": []}

        if plan.kind == "comparison":
            res = [self._resolve(p, fetched, traits) for p in plan.parts]
            if any(r.value is None for r in res):
                return missing()
            out = conflicted(res)
            if out:
                return out
            nums = [as_number(r.value or "") for r in res]
            if None in nums:
                return missing()
            pick_max = plan.direction == "max"
            i = 0 if (nums[0] >= nums[1]) == pick_max else 1  # type: ignore[operator]
            if noisy:
                i = 1 - i
            claims = [cl for r, e in zip(res, plan.entities) for cl in claims_for(r, e)]
            return answered(plan.entities[i], claims)

        if plan.kind == "two_hop":
            first = self._resolve(plan.parts[0], fetched, traits)
            if first.value is None:
                return missing()
            outer_q = analyse_question(q.text.replace(plan.span, first.value))
            second = self._resolve(outer_q, fetched, traits)
            if second.value is None:
                return missing()
            out = conflicted([first, second])
            if out:
                return out
            value = second.value
            if noisy and second.runner_up is not None:
                value = second.runner_up.value
            claims = claims_for(first, plan.parts[0].text) + claims_for(second, outer_q.text)
            return answered(value, claims)

        res = self._resolve(q, fetched, traits)
        if res.value is None:
            return missing()
        out = conflicted([res])
        if out:
            return out
        if noisy and res.runner_up is not None:
            r = res.runner_up
            return answered(r.value, [claim(r.sentence, [r.doc_id])])
        return answered(res.value, claims_for(res, q.text))


__all__ = [
    "BRIDGE_PATTERNS",
    "DEFAULT_PHRASES",
    "SYNONYM_GROUPS",
    "MockResearchModel",
    "Plan",
    "PromptTraits",
    "analyse_question",
    "detect_traits",
    "extract_question",
    "instruction_text",
    "plan_question",
]
