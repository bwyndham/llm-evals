"""`MockProvider`: a rule-based stand-in for the model, for offline demo mode.

This proves the PLUMBING, not model quality. Nothing here is a language model:
the spec writer is a handful of regexes tuned to the asks that
`ask_renderer` produces, and the judge is word-overlap arithmetic. A pass or a
ROLLBACK from an offline run says the pipeline (rendering, running, parsing,
checks, stats, gate, report) is wired correctly end to end. It says nothing
about how a real model would score.

What makes it useful for that is that its output genuinely depends on the
prompt, so a prompt edit changes behaviour the way the seeded regressions
intend, and the gate has something real to catch:

Mock spec writer (any prompt that is not a judge prompt)
    1. The ask is read from between ``<request>`` and ``</request>`` in the user
       prompt (or the whole user prompt if those tags are missing).
    2. The ask is segmented into topic blocks using bundling markers
       (``1) topic: ...``, ``also, totally separate thing (topic): ...``,
       ``First, on topic: ...`` / ``Separately, on topic: ...``, meeting-notes
       topic headers). Greetings, sign-offs, urgency lines, asides, email
       pleasantries and meeting-notes logistics are dropped. Actor intros
       ("This one is for our bookkeepers.") set the block's actor.
    3. Each remaining sentence is a requirement, unless it is framed as
       background ("For context," / "FWIW," / "Worth knowing:") or reads as a
       narrative (first person singular, past events, "right now", "because",
       ...), in which case it is a *hint*: an implied requirement.
    4. The SYSTEM prompt decides how the ticket is shaped (see `PromptTraits`):
       - split bundles: one scenario per requirement; otherwise a single
         scenario holding only the first topic's requirements;
       - flag assumptions: hints, hedged wording and a missing actor become
         assumptions / open questions; otherwise hints are written as firm
         scenarios, a plausible access-control scenario is invented whenever
         the actor is missing or the wording is hedged, and both lists are [].
    5. Sampling noise: per (system, user) pair a thread-safe call counter seeds
       an RNG from sha256(system, user, counter); with probability `noise` one
       extracted requirement is dropped. Same call order per pair, same output.

Mock judge (the user prompt ends with a fenced ```json block that has "task")
    Answers the four tasks in docs/contracts.md with transparent heuristics,
    mostly content-word overlap (lowercased, stopwords dropped, a trailing "s"
    stripped): an atom is covered, and a claim supported, when at least 50% of
    its content words appear in the ticket text / cited documents.

Suite responder (a suite's own `mock:` in suite.yaml)
    Any other task needs its own stand-in. `MockProvider(responder=fn)` calls
    `fn(system=..., user=..., rng=...)` for every non-judge prompt and returns
    the dict it gives as the output JSON. `eval init` writes one for its
    example task.

Token counts are len(text) // 4, request ids are "mock-<n>", the stop reason
is "end_turn". The `model` argument is ignored.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import threading
from dataclasses import dataclass, field
from typing import Any

from llm_evals.runners.providers.base import ProviderResponse

# --- System prompt traits ------------------------------------------------------

# Phrases are matched case-insensitively on the whitespace-normalized system prompt.
# A trait is ON when any "on" phrase is present and no "off" phrase is, so a small
# rewording keeps the behaviour, and a prompt that explicitly asks for the opposite
# flips it.
SPLIT_ON_PHRASES: tuple[str, ...] = (
    r"separate scenarios",
    r"(?:its|their) own scenario",
    r"never merge two (?:different )?asks",
    r"one scenario (?:per|for each) (?:ask|request)",
)
SPLIT_OFF_PHRASES: tuple[str, ...] = (
    r"a single \W?scenario",
    r"keep it to one scenario",
    r"only one scenario",
    r"exactly one \W?scenario",
)
FLAG_ON_PHRASES: tuple[str, ...] = (
    r"write an assumption",
    r"record (?:it|them|this) (?:under|in|as) (?:the )?assumptions",
    r"(?:do not|don't|never) guess silently",
    r"ask in open_questions",
    r"name the user you assumed",
    r"instead of (?:quietly )?(?:adding|inventing) a requirement",
)
FLAG_OFF_PHRASES: tuple[str, ...] = (
    r"leave (?:this|it|them|both) as \[\]",
    r"fill the gaps? yourself",
    r"(?:sensible|plausible), specific requirements",
    r"not a list of questions",
)


def _has_any(text: str, patterns: tuple[str, ...]) -> list[str]:
    return [p for p in patterns if re.search(p, text, re.IGNORECASE)]


@dataclass(frozen=True)
class PromptTraits:
    """What the spec writer's system prompt instructs, as the mock reads it."""

    split_bundles: bool
    flag_assumptions: bool
    evidence: dict[str, list[str]] = field(default_factory=dict, compare=False)

    @classmethod
    def detect(cls, system: str | None) -> PromptTraits:
        text = re.sub(r"\s+", " ", system or "")
        split_on, split_off = _has_any(text, SPLIT_ON_PHRASES), _has_any(text, SPLIT_OFF_PHRASES)
        flag_on, flag_off = _has_any(text, FLAG_ON_PHRASES), _has_any(text, FLAG_OFF_PHRASES)
        return cls(
            split_bundles=bool(split_on) and not split_off,
            flag_assumptions=bool(flag_on) and not flag_off,
            evidence={"split_on": split_on, "split_off": split_off, "flag_on": flag_on, "flag_off": flag_off},
        )


# --- Ask parsing ----------------------------------------------------------------

_REQUEST_TAG = re.compile(r"<request>\s*(.*?)\s*</request>", re.DOTALL | re.IGNORECASE)

URGENCY = re.compile(
    r"\burgent\b|\basap\b|as soon as possible|top priority|drop everything|no excuses|"
    r"breathing down|under a lot of pressure|\bthe ceo\b|please hurry|!!!|:rotating_light:",
    re.IGNORECASE,
)
ASIDE_START = re.compile(
    r"^(?:side note|unrelated|random\b|by the way|btw\b|funny story|heads up\b|i keep meaning to ask|"
    r"sorry (?:this|for)|we got sidetracked|also,? i'?m out|off topic|on another note)",
    re.IGNORECASE,
)
ASIDE_WORDS = re.compile(
    r"\b(?:coffee machine|my cat|wifi|wi-fi|parking|book club|pizza|wedding|neighbou?r|offsite photos|lunch)\b",
    re.IGNORECASE,
)
FILLER = re.compile(
    r"^(?:hope (?:you'?re|you are|all is|everyone is) (?:well|good)|quick (?:request|one|question) from|"
    r"writing (?:this )?up something|just a quick note)",
    re.IGNORECASE,
)
NOTES_META = re.compile(r"\bopened by saying\b|\bto write this up\b", re.IGNORECASE)
ACTOR_INTRO = [
    re.compile(r"^this (?:one )?(?:is for|comes from) (?:our |the )?(?P<a>[\w -]+?)\.?$", re.IGNORECASE),
    re.compile(r"^asking on behalf of (?:our |the )?(?P<a>[\w -]+?)\.?$", re.IGNORECASE),
    re.compile(r"^(?:our |the )?(?P<a>[\w -]+?) (?:keep|keeps|have been|are) asking for this\.?$", re.IGNORECASE),
    re.compile(r"^requested by (?:our |the )?(?P<a>[\w -]+?)\.?$", re.IGNORECASE),
]
HINT_FRAME = re.compile(
    r"^(?:for context|for background|fwiw|worth knowing|good to know|context|background)\s*[:,]\s*",
    re.IGNORECASE,
)
# Narrative cues: the sentence describes the world or a past event, not a wanted behaviour.
NARRATIVE = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\b(?:i|i'm|i've|i'd|my|me)\b",  # first person singular
        r"\blast (?:month|year|time|spring|summer|autumn|fall|winter|week|quarter'?s?)\b",
        r"\b(?:ago|right now|at month end|every single month|on bad terms|because)\b",
        r"\b(?:we|they|someone|somebody|people|nobody|everyone|our(?: \w+){1,2})\s+"
        r"(?:had|got|were|was|spent|ended|bumped|rebranded|complained|flagged|told|left|"
        r"screenshotted|kept|moved|opened|already)\b",
        r"\bwe(?:'ve| have) been\b|\bwe'?d like\b|\bwe don'?t want\b|\bwe have (?:hundreds|thousands)\b",
        r"\b(?:most|some|half|one|none) of (?:our|my|the)\b|\bhalf the\b|\bsome people\b|\bthe folks\b",
        r"\bnobody here\b|\bpeople (?:hate|lose|love|bounce)\b|\bwon'?t\b|\binsists?\b",
        r"\bkeep \w+ing\b|\b(?:told|emailing|asking) us\b|\bso \w+ that\b",
        r"\b(?:auditors?|reviewers?|advisors?|processors?|contracts)\b",
    )
]
MODAL = re.compile(
    r"\b(?:need|needs|should|shouldn't|must|can|can't|cannot|could|able|has to|have to|want|wants)\b",
    re.IGNORECASE,
)
HEDGE = re.compile(
    r"\bbasically\b|\bsome way\b|\bif that makes sense\b|\bit'?d help\b|\bit would really help\b|"
    r"\bcan we make it so\b|\bsort of\b|\bkind of\b|\bor something\b",
    re.IGNORECASE,
)
CONNECTIVE = re.compile(
    r"^(?:also,?|additionally,?|plus,?|oh,? and|and|note that|keep in mind that|"
    r"one more thing:|another thing:|finally,?)\s+",
    re.IGNORECASE,
)
TRAILING_HEDGE = re.compile(r",?\s*if that makes sense\s*$", re.IGNORECASE)

# Requirement framings -> (subject, phrase). Subject None means "the user" (unstated).
_ACTION_FRAMES = [
    re.compile(r"^(?:please )?make it so (?P<s>.+?) can (?P<p>.+)$", re.IGNORECASE),
    re.compile(r"^can we make it so (?P<s>.+?) can (?P<p>.+)$", re.IGNORECASE),
    re.compile(r"^it(?:'d| would) (?:help a lot|really help) if (?P<s>.+?) could (?P<p>.+)$", re.IGNORECASE),
    re.compile(r"^we need (?P<s>(?!to\b|some\b|a way\b).+?) to be able to (?P<p>.+)$", re.IGNORECASE),
    re.compile(
        r"^(?:we need to be able to|we have to be able to|we need a way to|we need some way to|"
        r"there needs to be a way to|it should be possible to|it needs to be possible to|"
        r"(?:please )?make it possible to|we basically need to)\s+(?P<p>.+)$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^(?P<s>[\w' -]+?) (?:need to|have to|must|should|needs to) be able to (?P<p>.+)$", re.IGNORECASE
    ),
    re.compile(r"^(?P<s>[\w' -]+?) (?:basically need to|need some way to|needs some way to) (?P<p>.+)$",
               re.IGNORECASE),
]
_UNSTATED_SUBJECTS = {"we", "you", "us", "someone", "people", "everyone"}

_BLOCK_MARKERS = [
    re.compile(r"^\s*\d+[).]\s*(?P<t>[^:]{1,80}):\s*(?P<rest>.*)$", re.IGNORECASE),
    re.compile(r"^\s*(?:also,?\s*)?(?:a\s+)?(?:totally |completely )?separate (?:thing|ask|request)\s*"
               r"\((?P<t>[^)]{1,80})\):\s*(?P<rest>.*)$", re.IGNORECASE),
    re.compile(r"^\s*(?:first|firstly),? on (?P<t>[^:]{1,80}):\s*(?P<rest>.*)$", re.IGNORECASE),
    re.compile(r"^\s*(?:separately|secondly|second|next),? on (?P<t>[^:]{1,80}):\s*(?P<rest>.*)$",
               re.IGNORECASE),
    re.compile(r"^\s*on a separate note,? regarding (?P<t>[^:]{1,80}):\s*(?P<rest>.*)$", re.IGNORECASE),
]
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9:@($\"'])")
_GREETING_LINE = re.compile(r"(?<![\w@])(?:hey|hi|hello|morning|afternoon|@here|@channel|dear)\b", re.IGNORECASE)
_CLOSER_LINE = re.compile(
    r"^(?:thanks|thank you|many thanks|ty|thx|cheers|lmk|best|kind regards|regards)\b", re.IGNORECASE
)
_SUBJECT_PREFIX = re.compile(
    r"^(?:(?:urgent)\W*\s*)?(?:request:|feature request\s*-|ask re:|re:|two requests:)?\s*", re.IGNORECASE
)


@dataclass
class Requirement:
    text: str  # the cleaned sentence (requirement wording kept verbatim)
    subject: str | None  # who the sentence names, if anyone
    phrase: str  # the verb phrase (action) or the full clause
    action: bool
    hedged: bool


@dataclass
class Block:
    topic: str | None
    actor: str | None = None
    requirements: list[Requirement] = field(default_factory=list)
    hints: list[str] = field(default_factory=list)


def extract_ask(user: str) -> str:
    match = _REQUEST_TAG.search(user)
    return match.group(1) if match else user.strip()


def _split_topics(lines: list[str], default_topic: str | None) -> list[tuple[str | None, str]]:
    """Group lines into (topic, text) blocks at bundling markers."""
    blocks: list[tuple[str | None, list[str]]] = [(default_topic, [])]
    for line in lines:
        for marker in _BLOCK_MARKERS:
            m = marker.match(line)
            if m:
                if not blocks[-1][1] and blocks[-1][0] == default_topic and len(blocks) == 1:
                    blocks = []
                blocks.append((m.group("t").strip(), [m.group("rest")]))
                break
        else:
            blocks[-1][1].append(line)
    return [(t, " ".join(ls).strip()) for t, ls in blocks if " ".join(ls).strip()]


def segment_ask(ask: str) -> list[tuple[str | None, list[str]]]:
    """Split an ask into topic blocks of sentences, dropping the wrapper (greeting, sign-off...)."""
    lines = [ln.rstrip() for ln in ask.strip().splitlines()]
    if lines and lines[0].lower().startswith("subject:"):
        return _segment_email(lines)
    if any(re.match(r"^\s*(?:attendees|present)\s*:", ln, re.IGNORECASE) for ln in lines):
        return _segment_notes(lines)
    return _segment_chat(lines)


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]


def _segment_email(lines: list[str]) -> list[tuple[str | None, list[str]]]:
    subject = _SUBJECT_PREFIX.sub("", lines[0].split(":", 1)[1].strip()).strip()
    body = "\n".join(lines[1:])
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
    kept: list[str] = []
    for p in paragraphs:
        first = p.splitlines()[0]
        if re.match(r"^(?:hi|hello|hey|dear|good morning)\b.*[,!]?$", first, re.IGNORECASE) and len(first.split()) <= 5:
            continue
        if _CLOSER_LINE.match(first) and len(first.split()) <= 4:
            continue  # sign-off paragraph (with the name under it)
        kept.append(" ".join(p.splitlines()))
    blocks = _split_topics(kept, subject or None)
    return [(t, _sentences(text)) for t, text in blocks]


def _segment_notes(lines: list[str]) -> list[tuple[str | None, list[str]]]:
    blocks: list[tuple[str | None, list[str]]] = []
    current: tuple[str | None, list[str]] | None = None
    for idx, raw in enumerate(lines):
        line = raw.strip()
        if not line or idx == 0 or re.match(r"^(?:attendees|present)\s*:", line, re.IGNORECASE):
            continue
        if re.match(r"^(?:action items|next steps|todo|to do)\s*:?$", line, re.IGNORECASE):
            break
        if line.startswith(("-", "*", "•")):
            sentence = line.lstrip("-*• ").strip()
            if current is None:
                current = (None, [])
                blocks.append(current)
            current[1].extend(_sentences(sentence))
            continue
        topic = re.sub(r"^(?:topic|discussed|agenda item|item)\s*:\s*", "", line, flags=re.IGNORECASE)
        current = (topic, [])
        blocks.append(current)
    # Drop blocks that only held wrapper bullets (e.g. an urgency opener before the first header).
    return [b for b in blocks if b[0] is not None or any(not URGENCY.search(s) for s in b[1])]


def _segment_chat(lines: list[str]) -> list[tuple[str | None, list[str]]]:
    kept: list[str] = []
    for idx, raw in enumerate(lines):
        line = raw.strip()
        if not line:
            continue
        words = re.sub(r"[^\w@' ]", " ", URGENCY.sub(" ", line)).split()
        if idx == 0 and _GREETING_LINE.search(line) and len(words) <= 6:
            continue
        if _CLOSER_LINE.match(line) and len(words) <= 6:
            continue
        if line.endswith(":") and len(words) <= 4:
            continue  # "two things:", "couple of asks:"
        kept.append(line)
    return [(t, _sentences(text)) for t, text in _split_topics(kept, None)]


def is_narrative(sentence: str) -> bool:
    return any(p.search(sentence) for p in NARRATIVE)


def _clean(sentence: str) -> str:
    s = sentence.strip()
    for _ in range(3):
        s = CONNECTIVE.sub("", s)
    s = TRAILING_HEDGE.sub("", s.rstrip(".!? "))
    return s.rstrip(".!? ").strip()


def parse_requirement(sentence: str) -> Requirement:
    hedged = bool(HEDGE.search(sentence))
    text = _clean(sentence)
    body = re.sub(r"\b(?:basically|just)\s+", "", text, flags=re.IGNORECASE) if hedged else text
    for frame in _ACTION_FRAMES:
        m = frame.match(body)
        if m:
            subject = (m.groupdict().get("s") or "").strip() or None
            if subject and subject.lower() in _UNSTATED_SUBJECTS:
                subject = None
            return Requirement(text=text, subject=subject, phrase=m.group("p").strip(), action=True, hedged=hedged)
    clause = text[:1].upper() + text[1:]
    return Requirement(text=text, subject=None, phrase=clause, action=False, hedged=hedged)


def classify(sentence: str) -> tuple[str, str | None]:
    """('drop' | 'actor' | 'hint' | 'requirement', detail)."""
    s = sentence.strip()
    if not s or URGENCY.search(s) or NOTES_META.search(s) or FILLER.match(s):
        return "drop", None
    if ASIDE_START.match(s) or ASIDE_WORDS.search(s):
        return "drop", None
    for pattern in ACTOR_INTRO:
        m = pattern.match(s)
        if m and not MODAL.search(s.replace("asking", "")):
            return "actor", m.group("a").strip()
    framed = HINT_FRAME.match(s)
    if framed:
        return "hint", _clean(s[framed.end():])
    if is_narrative(s):
        return "hint", _clean(s)
    return "requirement", None


def parse_ask(ask: str) -> list[Block]:
    blocks: list[Block] = []
    for topic, sentences in segment_ask(ask):
        block = Block(topic=topic)
        for sentence in sentences:
            kind, detail = classify(sentence)
            if kind == "actor":
                block.actor = detail
            elif kind == "hint" and detail:
                block.hints.append(detail)
            elif kind == "requirement":
                req = parse_requirement(sentence)
                if req.phrase:
                    block.requirements.append(req)
        if block.requirements or block.hints:
            blocks.append(block)
    return blocks


# --- Ticket building --------------------------------------------------------------

# Invented requirements for the no-assumption-flags behaviour. Written to share no
# keyword (and no content word) with any atom in suites/spec-writing/data/atoms.jsonl;
# tests/offline checks that.
INVENTED_SCENARIOS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "Restrict this area to staff with elevated rights",
        ("Given a staff user lacking elevated rights", "When they try to enter this area",
         'Then they hit a "not allowed" notice', "And the attempt is written to the security trail"),
    ),
    (
        "Ask for a fresh sign-in ahead of risky operations",
        ("Given a staff user with a stale sign-in", "When they start a risky operation",
         "Then they must sign in again"),
    ),
)


def _cap(text: str) -> str:
    return text[:1].upper() + text[1:]


def _short(text: str, limit: int) -> str:
    text = text.strip().rstrip(".")
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0]
    return cut.rstrip(",;: ")


def _who(block: Block) -> str:
    return f"one of the {block.actor}" if block.actor else "a user"


def _subject(req: Requirement) -> str:
    s = req.subject or "they"
    return s[:1].lower() + s[1:] if s[1:2].islower() else s


def _clause(req: Requirement) -> str:
    p = req.phrase
    return p[:1].lower() + p[1:] if p[1:2].islower() or p[1:2] == " " else p


def _label(block: Block) -> str:
    if block.topic:
        return block.topic
    first = block.requirements[0].phrase if block.requirements else (block.hints[0] if block.hints else "")
    return _short(first, 40)


def _scenario_lines(name: str, steps: list[str]) -> list[str]:
    return [f"  Scenario: {name}", *(f"    {s}" for s in steps), ""]


def _steps_for(req: Requirement, block: Block) -> list[str]:
    who = _who(block)
    if req.action:
        return [f"Given {who} is signed in", f"When they try to {req.phrase}", f"Then {_subject(req)} can {req.phrase}"]
    where = block.topic or "the product"
    return [f"Given {who} is using {where}", "When the rule applies", f"Then {_clause(req)}"]


def _requirement_scenarios(req: Requirement, block: Block) -> list[str]:
    return _scenario_lines(_cap(_short(req.phrase, 70)), _steps_for(req, block))


def build_ticket(blocks: list[Block], traits: PromptTraits, rng: random.Random) -> dict[str, Any]:
    if not traits.split_bundles and len(blocks) > 1:
        blocks = blocks[:1]  # one focused scenario: everything after the first topic is lost
    topics = [_label(b) for b in blocks if _label(b)]
    all_reqs = [r for b in blocks for r in b.requirements]
    title = _cap(_short(" and ".join(topics), 80)) or "Requested change"
    actors = [b.actor for b in blocks if b.actor]
    missing_actor = any(b.actor is None for b in blocks)
    hedged = any(r.hedged for r in all_reqs)

    lines = [f"Feature: {title}", ""]
    assumptions: list[str] = []
    questions: list[str] = []

    if traits.split_bundles:
        for block in blocks:
            for req in block.requirements:
                lines += _requirement_scenarios(req, block)
    else:
        block = blocks[0] if blocks else Block(topic=None)
        reqs = block.requirements
        if reqs:
            steps = _steps_for(reqs[0], block)
            for req in reqs[1:]:
                steps.append("And " + (f"{_subject(req)} can {req.phrase}" if req.action else _clause(req)))
            lines += _scenario_lines(_cap(_short(block.topic or reqs[0].phrase, 70)), steps)

    if traits.flag_assumptions:
        for block in blocks:
            assumptions += [f"The request implies, without stating it: {h}" for h in block.hints]
            if block.actor is None:
                where = f" for {block.topic}" if block.topic else ""
                assumptions.append(f"Assumed the user{where} is a signed-in account user; the request does not say.")
                questions.append(f"Who is the user{where}?")
        for req in all_reqs:
            if req.hedged:
                assumptions.append(f"Read the loosely worded ask as a firm requirement: {req.text}")
    else:
        for block in blocks:  # implied needs written up as firm criteria
            for hint in block.hints:
                lines += _scenario_lines(_cap(_short(hint, 70)), [
                    f"Given {_who(block)} is using {block.topic or 'the product'}",
                    "When the situation comes up", f"Then {hint}"])
        if missing_actor or hedged:
            invented_name, invented_steps = INVENTED_SCENARIOS[rng.randrange(len(INVENTED_SCENARIOS))]
            lines += _scenario_lines(invented_name, list(invented_steps))

    if len(lines) == 2:  # nothing extracted at all: still a valid, honest ticket
        lines += _scenario_lines("Request is clarified", [
            "Given the request as written", "When the team reviews it", "Then the requirements are agreed"])
        if traits.flag_assumptions:
            questions.append("What exactly is being asked for?")

    scope = ", ".join(topics) if topics else "the requested change"
    if traits.flag_assumptions:
        who = ", ".join(actors) if actors else "an unstated user (see assumptions)"
        description = f"This ticket covers {scope}. It is for {who}."
    else:
        who = ", ".join(actors) if actors and not missing_actor else "account users"
        description = f"This ticket covers {scope}. It is for {who}, who need it to work quickly and reliably."
    return {
        "title": title,
        "description": description,
        "acceptance_criteria": "\n".join(lines).rstrip() + "\n",
        "assumptions": assumptions,
        "open_questions": questions,
    }


# --- Mock judge -------------------------------------------------------------------

STOPWORDS = frozenset(
    """a an the and or but if then else of to in on at by for from with without into onto over under
    up down out off as is are was were be been being am do does did done has have had having can could
    should would will shall may might must need needs not no nor so than that this these those it its
    they them their there here who whom whose which what when where why how all any each every some
    such only own same other more most less few many much very just also too again once ever never
    we us our you your he she his her him i me my mine able given user users one two per via""".split()
)
_WORD = re.compile(r"[a-z0-9$%+]+(?:[-'.,][a-z0-9%+]+)*")


def content_words(text: str) -> set[str]:
    words = set()
    for w in _WORD.findall(text.lower()):
        w = w.strip("'.,")
        if w.endswith("'s"):
            w = w[:-2]
        if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
            w = w[:-1]
        if w and w not in STOPWORDS:
            words.add(w)
    return words


def overlap(text: str, reference: str) -> tuple[float, list[str]]:
    words = content_words(text)
    if not words:
        return 0.0, []
    ref = content_words(reference)
    hit = sorted(words & ref)
    return len(hit) / len(words), hit


def _ticket_text(ticket: dict[str, Any]) -> str:
    return "\n".join(str(ticket.get(k) or "") for k in ("title", "description", "acceptance_criteria"))


def ticket_scenarios(ticket: dict[str, Any]) -> list[str]:
    """Scenario blocks (name + steps) by a plain line scan of acceptance_criteria."""
    text = str(ticket.get("acceptance_criteria") or "")
    chunks: list[list[str]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if re.match(r"^(?:scenario(?: outline| template)?|example)\s*:", stripped, re.IGNORECASE):
            chunks.append([stripped.split(":", 1)[1].strip()])
        elif chunks and re.match(r"^(?:given|when|then|and|but|\*)\b", stripped, re.IGNORECASE):
            chunks[-1].append(re.sub(r"^(?:given|when|then|and|but|\*)\s*", "", stripped, flags=re.IGNORECASE))
    return ["\n".join(c) for c in chunks]


def _judge_faithfulness(payload: dict[str, Any]) -> dict[str, Any]:
    ticket = payload.get("ticket") or {}
    text = _ticket_text(ticket)
    atoms_out = []
    atom_words: set[str] = set()
    for atom in payload.get("atoms") or []:
        share, hit = overlap(str(atom.get("text", "")), text)
        atom_words |= content_words(str(atom.get("text", "")))
        covered = share >= 0.5
        evidence = (f"{len(hit)} of {len(content_words(str(atom.get('text', ''))))} content words in the ticket: "
                    + ", ".join(hit)) if hit else "no content words of this atom appear in the ticket"
        atoms_out.append({"id": str(atom.get("id")), "covered": covered, "evidence": evidence})
    invented = []
    for scenario in ticket_scenarios(ticket):
        if not content_words(scenario) & atom_words:
            invented.append({"text": scenario.splitlines()[0],
                             "reason": "shares no content word with any requirement atom"})
    return {"atoms": atoms_out, "invented": invented}


def _judge_checklist(payload: dict[str, Any]) -> dict[str, Any]:
    ticket = payload.get("ticket") or payload.get("output") or {}
    ask = str(payload.get("ask") or "")
    blocks = parse_ask(ask) if ask else []
    n_topics = max(1, len(blocks))
    n_scenarios = len(ticket_scenarios(ticket))
    ac = str(ticket.get("acceptance_criteria") or "")
    flagged = bool(ticket.get("assumptions") or ticket.get("open_questions"))
    missing_actor = any(b.actor is None for b in blocks)
    items = []
    for item in payload.get("items") or []:
        q = str(item.get("question", "")).lower()
        if re.search(r"bundl|separate (?:ask|request)|own scenario|split", q):
            ok = n_topics <= 1 or n_scenarios >= n_topics
            reason = f"the ask has {n_topics} topic(s) and the ticket has {n_scenarios} scenario(s)"
        elif re.search(r"assum|open question|who the user|actor|unstated|guess", q):
            ok = flagged or not missing_actor
            reason = ("the ask names its user" if not missing_actor
                      else "the ask omits the user and the ticket " + ("flags it" if flagged else "does not flag it"))
        elif re.search(r"title|80 char", q):
            ok = 0 < len(str(ticket.get("title") or "")) <= 80
            reason = f"title is {len(str(ticket.get('title') or ''))} chars"
        elif re.search(r"urgen|pressure|asap", q):
            ok = not URGENCY.search(_ticket_text(ticket))
            reason = "no urgency language in the ticket" if ok else "the ticket repeats urgency language"
        elif re.search(r"given|gherkin|testable|scenario", q):
            ok = all(re.search(rf"^\s*{k}\b", ac, re.IGNORECASE | re.MULTILINE) for k in ("given", "when", "then"))
            reason = "scenarios use Given/When/Then" if ok else "Given/When/Then steps are missing"
        else:
            ok, reason = True, "no rule for this item; the mock judge answers yes"
        items.append({"id": str(item.get("id")), "answer": "yes" if ok else "no", "reason": reason})
    return {"items": items}


def _judge_pairwise(payload: dict[str, Any]) -> dict[str, Any]:
    def score(ticket: Any) -> int:
        ticket = ticket if isinstance(ticket, dict) else {}
        return len(ticket_scenarios(ticket)) + int(bool(ticket.get("assumptions") or ticket.get("open_questions")))

    a, b = score(payload.get("a")), score(payload.get("b"))
    winner = "a" if a > b else "b" if b > a else "tie"
    return {"winner": winner, "reason": f"score = scenarios + 1 if it flags assumptions: a={a}, b={b}"}


def _judge_claim_support(payload: dict[str, Any]) -> dict[str, Any]:
    out = []
    for claim in payload.get("claims") or []:
        docs = claim.get("cited_docs") or []
        reference = "\n".join(str(d.get("text", "")) for d in docs if isinstance(d, dict))
        share, hit = overlap(str(claim.get("text", "")), reference)
        supported = bool(docs) and share >= 0.5
        reason = ("no cited documents" if not docs
                  else f"{share:.0%} of the claim's content words appear in the cited text")
        out.append({"id": str(claim.get("id")), "supported": supported, "reason": reason})
    return {"claims": out}


JUDGE_TASKS = {
    "faithfulness": _judge_faithfulness,
    "checklist": _judge_checklist,
    "pairwise": _judge_pairwise,
    "claim_support": _judge_claim_support,
}


def judge_payload(user: str) -> dict[str, Any] | None:
    """The payload of the fenced json block that ENDS the prompt, if it has a "task" key."""
    start = max(user.rfind("```json"), user.rfind("```JSON"))
    if start < 0:
        return None
    rest = user[start + len("```json"):]
    end = rest.find("```")
    if end < 0 or rest[end + 3:].strip():
        return None
    try:
        payload = json.loads(rest[:end])
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) and "task" in payload else None


# --- Provider -----------------------------------------------------------------------


class MockProvider:
    """Text `Provider` that answers offline: a rule-based spec writer and a mock judge.

    Proves the plumbing, not model quality (see the module docstring).

    - `noise`: per-call probability of dropping one extracted requirement, so
      A/A runs vary. Deterministic for a given call order per (system, user) pair.
    - `seed`: mixed into every RNG, to get a different but reproducible stream.
    """

    def __init__(self, *, noise: float = 0.05, seed: int = 0,
                 responder: Any | None = None) -> None:
        self.responder = responder
        if not 0.0 <= noise <= 1.0:
            raise ValueError(f"noise must be in [0, 1], got {noise!r}")
        self.noise = noise
        self.seed = seed
        self._lock = threading.Lock()
        self._pair_calls: dict[str, int] = {}
        self._requests = 0

    def _next(self, system: str | None, user: str) -> tuple[random.Random, int]:
        pair = hashlib.sha256(f"{system or ''}\0{user}".encode()).hexdigest()
        with self._lock:
            counter = self._pair_calls.get(pair, 0)
            self._pair_calls[pair] = counter + 1
            self._requests += 1
            request_n = self._requests
        digest = hashlib.sha256(f"{self.seed}\0{system or ''}\0{user}\0{counter}".encode()).digest()
        return random.Random(int.from_bytes(digest[:8], "big")), request_n

    def write_spec(self, system: str | None, user: str, rng: random.Random) -> dict[str, Any]:
        traits = PromptTraits.detect(system)
        blocks = parse_ask(extract_ask(user))
        if rng.random() < self.noise:
            slots = [(i, j) for i, b in enumerate(blocks) for j in range(len(b.requirements))]
            if len(slots) > 1:
                i, j = slots[rng.randrange(len(slots))]
                del blocks[i].requirements[j]
        return build_ticket(blocks, traits, rng)

    def complete(
        self, *, model: str, system: str | None, user: str, params: dict[str, Any]
    ) -> ProviderResponse:
        rng, n = self._next(system, user)
        payload = judge_payload(user)
        if payload is not None:
            handler = JUDGE_TASKS.get(str(payload.get("task")))
            body: dict[str, Any] = (
                handler(payload) if handler else {"error": f"mock judge: unknown task {payload.get('task')!r}"}
            )
        elif self.responder is not None:
            body = self.responder(system=system, user=user, rng=rng)
        else:
            body = self.write_spec(system, user, rng)
        text = json.dumps(body, ensure_ascii=False)
        return ProviderResponse(
            text=text,
            input_tokens=(len(system or "") + len(user)) // 4,
            output_tokens=len(text) // 4,
            request_id=f"mock-{n}",
            stop_reason="end_turn",
        )


__all__ = [
    "INVENTED_SCENARIOS",
    "MockProvider",
    "PromptTraits",
    "content_words",
    "judge_payload",
    "parse_ask",
]
