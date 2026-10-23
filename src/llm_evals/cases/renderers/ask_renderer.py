"""`ask_renderer`: turn requirement atoms into a messy ask with controlled features.

Cases are constructed from a known answer. An atom set (the true requirements)
is rendered into a Slack message, email or meeting notes whose features are
controlled and recorded in `case.strata`:

- `vagueness`: `explicit` states every atom; `vague` paraphrases each loosely
  (keeping its key nouns and numbers); `hinted` states every atom except one per
  set, which is only implied by a remark (see `source["hinted_atom_ids"]`).
- `bundling`: `single` uses one atom set; `bundled` packs two sets into one ask.
- `tone`: `neutral`, `urgent` (pressure that must not change scope or
  priority) or `rambling` (irrelevant asides).
- `missing_actor`: `yes` leaves out who the user is.
- `source_format`: `slack`, `email` or `meeting_notes`.

Rendering is template-based and deterministic for a given seed. No model is
called. Rendering happens in two steps: a *plan* fixes how each atom is
realized (its mode and exact phrase, stored in `source["atoms"]`), then a
*composition* step wraps those phrases in framing sentences, greetings, tone
and format. `reword_ask` re-runs only the composition step with a new seed,
which is why it can change the wording without touching any atom's content.

Atom ids in `source` are qualified as ``"<set id>/<atom id>"`` so they stay
unique when two sets are bundled. Each atom also carries `set_id` and
`local_id` (the id in atoms.jsonl).
"""

from __future__ import annotations

import hashlib
import itertools
import json
import random
import re
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from llm_evals.registry import register_renderer
from llm_evals.types import Case

# --- Features ----------------------------------------------------------------

FEATURE_VALUES: dict[str, list[str]] = {
    "vagueness": ["explicit", "vague", "hinted"],
    "bundling": ["single", "bundled"],
    "tone": ["neutral", "urgent", "rambling"],
    "missing_actor": ["no", "yes"],
    "source_format": ["slack", "email", "meeting_notes"],
}
FEATURE_ORDER = list(FEATURE_VALUES)
DEFAULT_FEATURES: dict[str, str] = {name: values[0] for name, values in FEATURE_VALUES.items()}


def normalize_features(features: dict[str, Any] | None) -> dict[str, str]:
    """Fill defaults, stringify and validate feature values."""
    features = dict(features or {})
    unknown = sorted(set(features) - set(FEATURE_VALUES))
    if unknown:
        raise ValueError(f"unknown ask_renderer feature(s): {', '.join(unknown)}")
    out: dict[str, str] = {}
    for name in FEATURE_ORDER:
        value = str(features.get(name, DEFAULT_FEATURES[name]))
        if value not in FEATURE_VALUES[name]:
            allowed = " | ".join(FEATURE_VALUES[name])
            raise ValueError(f"feature {name}={value!r} is not one of {allowed}")
        out[name] = value
    return out


# --- Atom data ---------------------------------------------------------------


class AtomSpec(BaseModel):
    """One requirement atom as stored in atoms.jsonl.

    `text` is the canonical requirement (the ground truth). The renderer never
    shows `text` directly; it uses `say` (explicit wording), `vague` (loose
    wording) or `hint` (an indirect remark). None of those mention the actor,
    so `missing_actor=yes` can omit it. `form` says whether `say`/`vague` are
    verb phrases that need a subject ("action") or full clauses ("clause").
    `keywords` are lowercase key nouns/numbers present in both `say` and `vague`.
    """

    id: str
    text: str
    kind: Literal["functional", "constraint", "non_functional"]
    say: str
    vague: str
    keywords: list[str] = Field(min_length=1)
    form: Literal["action", "clause"] = "clause"
    hint: str | None = None


class AtomSetSpec(BaseModel):
    id: str
    domain: str
    actor: str
    topic: str
    atoms: list[AtomSpec] = Field(min_length=1)


def load_atom_sets(path: Path) -> list[AtomSetSpec]:
    """Load and validate atoms.jsonl. Raises ValueError with the line number on error."""
    sets: list[AtomSetSpec] = []
    seen: set[str] = set()
    for lineno, raw in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            spec = AtomSetSpec.model_validate(json.loads(raw))
        except Exception as exc:  # noqa: BLE001 - re-raised with location
            raise ValueError(f"{path}:{lineno}: invalid atom set: {exc}") from exc
        if spec.id in seen:
            raise ValueError(f"{path}:{lineno}: duplicate atom set id {spec.id!r}")
        local_ids = [a.id for a in spec.atoms]
        if len(set(local_ids)) != len(local_ids):
            raise ValueError(f"{path}:{lineno}: duplicate atom ids in set {spec.id!r}")
        seen.add(spec.id)
        sets.append(spec)
    return sets


# --- Seeds and ids -----------------------------------------------------------


def _derive_int(*parts: Any) -> int:
    blob = json.dumps(parts, sort_keys=True, default=str).encode("utf-8")
    return int.from_bytes(hashlib.sha256(blob).digest()[:8], "big")


def case_id_for(atom_set_ids: list[str], features: dict[str, str], seed: int) -> str:
    """Stable short id from (atom set ids, features, seed)."""
    blob = json.dumps(
        {"atom_set_ids": list(atom_set_ids), "features": features, "seed": seed},
        sort_keys=True,
    ).encode("utf-8")
    return "ask-" + hashlib.sha256(blob).hexdigest()[:12]


# --- Phrasing tables -----------------------------------------------------------

# Framing for "action" atoms (verb phrases). {subj}/{Subj} is the actor.
ACTION_EXPLICIT_ACTOR = [
    "{Subj} need to be able to {p}.",
    "{Subj} should be able to {p}.",
    "We need {subj} to be able to {p}.",
    "Please make it so {subj} can {p}.",
    "{Subj} must be able to {p}.",
]
ACTION_EXPLICIT_NO_ACTOR = [
    "We need to be able to {p}.",
    "There needs to be a way to {p}.",
    "It should be possible to {p}.",
    "Please make it possible to {p}.",
]
ACTION_VAGUE_ACTOR = [
    "{Subj} basically need to {p}.",
    "It'd help a lot if {subj} could {p}.",
    "{Subj} need some way to {p}.",
    "Can we make it so {subj} can {p}?",
]
ACTION_VAGUE_NO_ACTOR = [
    "We basically need to {p}.",
    "It'd help a lot if we could {p}.",
    "We need some way to {p}.",
    "Can we make it so you can {p}?",
]
# Framing for "clause" atoms (full clauses).
CLAUSE_EXPLICIT = ["{P}.", "Also, {p}.", "Note that {p}.", "One more thing: {p}."]
CLAUSE_VAGUE = ["{P}.", "Oh and {p}.", "Plus {p}.", "Also {p}, if that makes sense."]
CLAUSE_LEAD = ["{P}."]  # first sentence of a block: no connective
HINT_FRAMES = ["{P}.", "For context, {p}.", "FWIW, {p}.", "Worth knowing: {p}."]

SUBJECTS = ["{actors}", "our {actors}", "the {actors}"]
ACTOR_INTROS = [
    "This one is for our {actors}.",
    "This comes from the {actors}.",
    "Asking on behalf of our {actors}.",
    "Our {actors} keep asking for this.",
]

ASIDES = [
    "Side note: the coffee machine on the third floor is broken again.",
    "Unrelated, but my cat has figured out how to open the pantry door.",
    "Also I'm out Friday for my cousin's wedding, so replies might be slow.",
    "Sorry this is all over the place, I've had back-to-back calls since 8.",
    "We got sidetracked at lunch arguing about the best pizza place in town, so this took a while to write up.",
    "Random, but has anyone else noticed the office wifi dropping every afternoon?",
    "By the way, the offsite photos are finally up on the shared drive.",
    "I keep meaning to ask: are we still doing the book club this month?",
    "Funny story, I tried to explain this to my neighbour over the weekend and lost them in about ten seconds.",
    "Heads up, the parking garage is closed for repairs next week.",
]

URGENT_OPENERS = ["URGENT!!!", ":rotating_light: URGENT :rotating_light:", "Need this ASAP!!!"]
URGENT_LINES = [
    "This is extremely urgent and needs to go out ASAP.",
    "Leadership is breathing down my neck on this one, it's URGENT!!!",
    "Please treat this as top priority, we're under a lot of pressure here!!!",
]
URGENT_CLOSERS = [
    "This is super urgent, please drop everything!!!",
    "Need this shipped by end of week, no excuses!!!",
    "The CEO is asking about this every day, please hurry!!!",
]

NAMES = ["Priya", "Marcus", "Dana", "Tomás", "Aiko", "Sam", "Leila", "Jonah", "Grace", "Femi"]

SLACK_GREETINGS = ["hey team", "hi all", "@here quick one", "hey folks", "morning all"]
SLACK_BUNDLE_INTROS = ["two things:", "couple of asks:", "two separate requests:"]
SLACK_CLOSERS = ["thanks!", "ty!", "lmk if anything's unclear", "thx all"]

EMAIL_GREETINGS = ["Hi team,", "Hello product team,", "Hi all,", "Hi everyone,"]
EMAIL_INTROS = ["Hope you're well.", "Quick request from our side.", "Writing up something that came up this week."]
EMAIL_SUBJECTS = ["{T}", "Request: {t}", "Feature request - {t}", "Ask re: {t}"]
EMAIL_BUNDLE_SUBJECTS = ["Two requests: {t1} and {t2}", "{T1} + {t2}", "A couple of asks ({t1}, {t2})"]
EMAIL_SIGNOFFS = ["Thanks,", "Best,", "Cheers,", "Thanks so much,"]

NOTES_TITLES = [
    "Product sync - notes",
    "Weekly ops meeting notes",
    "Notes from Tuesday's planning call",
    "Customer feedback review - notes",
]
NOTES_TOPIC_HEADERS = ["Topic: {t}", "{T}", "Discussed: {t}"]

# Fixed synonym table used by reword_ask on framing text only (never on atom phrases).
SYNONYMS: list[tuple[str, str]] = [
    ("need to be able to", "have to be able to"),
    ("There needs to be a way to", "We need a way to"),
    ("It should be possible to", "It needs to be possible to"),
    ("Please make it", "Make it"),
    ("It'd help a lot if", "It would really help if"),
    ("Note that", "Keep in mind that"),
    ("One more thing:", "Another thing:"),
    ("Also,", "Additionally,"),
    ("Oh and", "Oh, and"),
    ("For context,", "For background,"),
    ("Worth knowing:", "Good to know:"),
    ("Separately, on", "On a separate note, regarding"),
    ("hey team", "hi team"),
    ("hey folks", "hi folks"),
    ("quick one", "quick request"),
    ("two things:", "two requests:"),
    ("thanks!", "thank you!"),
    ("Thanks,", "Many thanks,"),
    ("Best,", "Kind regards,"),
    ("Hope you're well.", "Hope all is well."),
    ("Attendees:", "Present:"),
    ("Action items:", "Next steps:"),
    ("ASAP", "as soon as possible"),
]
_SYN_MAP = dict(SYNONYMS)
_SYN_RE = re.compile("|".join(re.escape(a) for a, _ in sorted(SYNONYMS, key=lambda s: -len(s[0]))))


def _swap_synonyms(text: str) -> str:
    return _SYN_RE.sub(lambda m: _SYN_MAP[m.group(0)], text)


# --- Composition ---------------------------------------------------------------


def _cap(s: str) -> str:
    return s[:1].upper() + s[1:]


def _plural(actor: str) -> str:
    return actor if actor.endswith("s") else actor + "s"


class _Slots:
    """Holds content strings (atom phrases, actors, topics, asides) out of the scaffold.

    Framing text is built with opaque tokens in place of content, so synonym
    swaps can never touch an atom's wording.
    """

    def __init__(self) -> None:
        self.values: list[str] = []

    def put(self, value: str) -> str:
        self.values.append(value)
        return f"\x00{len(self.values) - 1}\x00"

    def fill(self, text: str) -> str:
        return re.sub("\x00(\\d+)\x00", lambda m: self.values[int(m.group(1))], text)


def _realize(atom: dict[str, Any], actor: str | None, rng: random.Random, slots: _Slots, *, first: bool) -> str:
    phrase = atom["phrase"]
    mode = atom["mode"]
    if mode == "hinted":
        frame = rng.choice(CLAUSE_LEAD if first else HINT_FRAMES)
    elif atom["form"] == "action":
        if actor:
            frame = rng.choice(ACTION_EXPLICIT_ACTOR if mode == "explicit" else ACTION_VAGUE_ACTOR)
        else:
            frame = rng.choice(ACTION_EXPLICIT_NO_ACTOR if mode == "explicit" else ACTION_VAGUE_NO_ACTOR)
    elif first:
        frame = rng.choice(CLAUSE_LEAD)
    else:
        frame = rng.choice(CLAUSE_EXPLICIT if mode == "explicit" else CLAUSE_VAGUE)

    values = {"p": slots.put(phrase), "P": slots.put(_cap(phrase))}
    if actor:
        subj = rng.choice(SUBJECTS).format(actors=_plural(actor))
        values["subj"] = slots.put(subj)
        values["Subj"] = slots.put(_cap(subj))
    return frame.format(**{k: v for k, v in values.items() if "{" + k + "}" in frame})


def _blocks(source: dict[str, Any], features: dict[str, str], rng: random.Random, slots: _Slots,
            *, reorder: bool) -> tuple[list[dict[str, Any]], list[list[str]]]:
    sets = list(source["sets"])
    if reorder and len(sets) > 1 and rng.random() < 0.5:
        sets.reverse()
    actor_stated = features["missing_actor"] == "no"
    blocks: list[list[str]] = []
    for s in sets:
        atoms = [a for a in source["atoms"] if a["set_id"] == s["id"]]
        if reorder:
            # Keep the lead atom first (later loose phrasings may say "it"), shuffle the rest.
            head, tail = atoms[:1], atoms[1:]
            rng.shuffle(tail)
            atoms = head + tail
        actor = s["actor"] if actor_stated else None
        sentences: list[str] = []
        if actor:
            sentences.append(rng.choice(ACTOR_INTROS).format(actors=slots.put(_plural(actor))))
        for i, atom in enumerate(atoms):
            sentences.append(_realize(atom, actor, rng, slots, first=(i == 0 and not sentences)))
        blocks.append(sentences)

    for aside in source.get("asides") or []:
        block = blocks[rng.randrange(len(blocks))]
        block.insert(rng.randint(1, len(block)), slots.put(aside))
    return sets, blocks


def _slack(sets: list[dict[str, Any]], blocks: list[list[str]], features: dict[str, str],
           rng: random.Random, slots: _Slots) -> str:
    urgent = features["tone"] == "urgent"
    greeting = rng.choice(SLACK_GREETINGS)
    first = f"{rng.choice(URGENT_OPENERS)} {greeting}" if urgent else greeting
    lines = [first + rng.choice([",", " -", ":"])]
    if len(blocks) == 1:
        lines.append(" ".join(blocks[0]))
    else:
        topics = [slots.put(s["topic"]) for s in sets]
        if rng.random() < 0.5:
            lines.append(rng.choice(SLACK_BUNDLE_INTROS))
            for i, (topic, block) in enumerate(zip(topics, blocks), 1):
                lines.append(f"{i}) {topic}: " + " ".join(block))
        else:
            lines.append(" ".join(blocks[0]))
            lines.append(f"also, totally separate thing ({topics[1]}): " + " ".join(blocks[1]))
    if urgent:
        lines.append(rng.choice(URGENT_CLOSERS))
    lines.append(rng.choice(SLACK_CLOSERS))
    return "\n".join(lines)


def _email(sets: list[dict[str, Any]], blocks: list[list[str]], features: dict[str, str],
           rng: random.Random, slots: _Slots) -> str:
    urgent = features["tone"] == "urgent"
    if len(sets) == 1:
        t = sets[0]["topic"]
        subject = rng.choice(EMAIL_SUBJECTS).format(t=slots.put(t), T=slots.put(_cap(t)))
    else:
        t1, t2 = sets[0]["topic"], sets[1]["topic"]
        subject = rng.choice(EMAIL_BUNDLE_SUBJECTS).format(
            t1=slots.put(t1), T1=slots.put(_cap(t1)), t2=slots.put(t2)
        )
    if urgent:
        subject = rng.choice(["URGENT: ", "URGENT!!! "]) + subject
    paragraphs = [rng.choice(EMAIL_GREETINGS)]
    if urgent:
        paragraphs.append(rng.choice(URGENT_LINES))
    elif rng.random() < 0.6:
        paragraphs.append(rng.choice(EMAIL_INTROS))
    if len(blocks) == 1:
        paragraphs.append(" ".join(blocks[0]))
    else:
        topics = [slots.put(s["topic"]) for s in sets]
        paragraphs.append(f"First, on {topics[0]}: " + " ".join(blocks[0]))
        paragraphs.append(f"Separately, on {topics[1]}: " + " ".join(blocks[1]))
    if urgent:
        paragraphs.append(rng.choice(URGENT_CLOSERS))
    paragraphs.append(f"{rng.choice(EMAIL_SIGNOFFS)}\n{rng.choice(NAMES)}")
    return f"Subject: {subject}\n\n" + "\n\n".join(paragraphs)


def _meeting_notes(sets: list[dict[str, Any]], blocks: list[list[str]], features: dict[str, str],
                   rng: random.Random, slots: _Slots) -> str:
    urgent = features["tone"] == "urgent"
    attendees = rng.sample(NAMES, 3)
    lines = [rng.choice(NOTES_TITLES), "Attendees: " + ", ".join(attendees), ""]
    if urgent:
        lines.append(f"- {attendees[0]} opened by saying this is URGENT!!! and has to ship ASAP")
    for s, block in zip(sets, blocks):
        header = rng.choice(NOTES_TOPIC_HEADERS)
        lines.append(header.format(t=slots.put(s["topic"]), T=slots.put(_cap(s["topic"]))))
        lines.extend(f"- {sentence}" for sentence in block)
        lines.append("")
    if urgent:
        lines.append(f"- {attendees[1]}: {rng.choice(URGENT_CLOSERS)}")
        lines.append("")
    lines.append("Action items:")
    lines.append(f"- {rng.choice(attendees)} to write this up for the product team")
    return "\n".join(lines)


_WRAPPERS = {"slack": _slack, "email": _email, "meeting_notes": _meeting_notes}


def _compose(source: dict[str, Any], features: dict[str, str], rng: random.Random,
             *, reorder: bool = False, swap: bool = False) -> str:
    slots = _Slots()
    sets, blocks = _blocks(source, features, rng, slots, reorder=reorder)
    scaffold = _WRAPPERS[features["source_format"]](sets, blocks, features, rng, slots)
    if swap:
        scaffold = _swap_synonyms(scaffold)
    return slots.fill(scaffold)


# --- Renderer ------------------------------------------------------------------


@register_renderer("ask_renderer")
class AskRenderer:
    """Renders atom sets from atoms.jsonl into messy asks (constructed cases)."""

    def __init__(self, atoms_path: Path) -> None:
        self.atoms_path = Path(atoms_path)
        self.atom_sets = load_atom_sets(self.atoms_path)
        self._by_id = {s.id: s for s in self.atom_sets}
        self._ids = sorted(self._by_id)
        if len(self._ids) < 2:
            raise ValueError("ask_renderer needs at least 2 atom sets (bundled cases use two)")

    def render(self, features: dict[str, Any], seed: int) -> Case:
        """Render one case; the atom set(s) are picked by `seed`."""
        feats = normalize_features(features)
        k = 2 if feats["bundling"] == "bundled" else 1
        set_ids = random.Random(_derive_int("pick", seed)).sample(self._ids, k)
        return self.render_sets(set_ids, feats, seed)

    def render_sets(self, atom_set_ids: list[str], features: dict[str, Any], seed: int) -> Case:
        """Render one case from the given atom set ids (2 ids for bundled, 1 for single)."""
        feats = normalize_features(features)
        expected_k = 2 if feats["bundling"] == "bundled" else 1
        if len(atom_set_ids) != expected_k or len(set(atom_set_ids)) != expected_k:
            raise ValueError(f"bundling={feats['bundling']} needs {expected_k} distinct atom set id(s)")
        specs = [self._by_id[i] for i in atom_set_ids]
        plan_rng = random.Random(_derive_int("plan", seed, list(atom_set_ids), feats))
        frame_rng = random.Random(_derive_int("frame", seed, list(atom_set_ids), feats))

        atoms: list[dict[str, Any]] = []
        hinted: list[str] = []
        for spec in specs:
            hinted_local: str | None = None
            if feats["vagueness"] == "hinted":
                candidates = [a.id for a in spec.atoms if a.hint]
                if candidates:
                    hinted_local = plan_rng.choice(candidates)
            for a in spec.atoms:
                if a.id == hinted_local:
                    mode, phrase = "hinted", a.hint or ""
                elif feats["vagueness"] == "vague":
                    mode, phrase = "vague", a.vague
                else:
                    mode, phrase = "explicit", a.say
                qid = f"{spec.id}/{a.id}"
                if mode == "hinted":
                    hinted.append(qid)
                atoms.append({
                    "id": qid,
                    "local_id": a.id,
                    "set_id": spec.id,
                    "text": a.text,
                    "kind": a.kind,
                    "keywords": list(a.keywords),
                    "form": a.form,
                    "mode": mode,
                    "phrase": phrase,
                })

        asides: list[str] = []
        if feats["tone"] == "rambling":
            asides = plan_rng.sample(ASIDES, 2 if expected_k == 1 else 3)

        source: dict[str, Any] = {
            "atoms": atoms,
            "actor": specs[0].actor,
            "actors": {s.id: s.actor for s in specs},
            "actor_stated": feats["missing_actor"] == "no",
            "atom_set_ids": list(atom_set_ids),
            "hinted_atom_ids": hinted,
            "sets": [{"id": s.id, "domain": s.domain, "actor": s.actor, "topic": s.topic} for s in specs],
            "asides": asides,
            "render": {"renderer": "ask_renderer", "seed": seed},
        }
        text = _compose(source, feats, frame_rng)
        return Case(
            id=case_id_for(list(atom_set_ids), feats, seed),
            input={"ask": text},
            source=source,
            strata=feats,
            provenance="constructed",
            tags=sorted({f"domain:{s.domain}" for s in specs}),
        )

    def render_suite(self, n: int, seed: int, features: dict[str, list[str]] | None = None) -> list[Case]:
        """Render `n` cases covering the feature grid in a balanced way.

        Every full copy of the grid that fits in `n` is used once; the remainder
        is drawn so each feature's values stay as balanced as possible. Atom
        sets are drawn from shuffled passes over all sets, so each set is used
        about equally often.
        """
        if n < 0:
            raise ValueError("n must be non-negative")
        features = features or {}
        unknown = sorted(set(features) - set(FEATURE_VALUES))
        if unknown:
            raise ValueError(f"unknown ask_renderer feature(s): {', '.join(unknown)}")
        values = {}
        for name in FEATURE_ORDER:
            vals = [str(v) for v in features.get(name, FEATURE_VALUES[name])]
            if not vals or any(v not in FEATURE_VALUES[name] for v in vals):
                raise ValueError(f"bad values for feature {name}: {vals}")
            values[name] = list(dict.fromkeys(vals))

        rng = random.Random(_derive_int("suite", seed))
        grid = [dict(zip(FEATURE_ORDER, combo)) for combo in itertools.product(*values.values())]
        combos = [dict(c) for c in grid] * (n // len(grid))
        remainder = n % len(grid)
        if remainder:
            columns = {}
            for name in FEATURE_ORDER:
                vals = list(values[name])
                rng.shuffle(vals)
                col = (vals * (remainder // len(vals) + 1))[:remainder]
                rng.shuffle(col)
                columns[name] = col
            combos += [{name: columns[name][i] for name in FEATURE_ORDER} for i in range(remainder)]
        rng.shuffle(combos)

        pool: list[str] = []

        def draw(k: int) -> list[str]:
            chosen: list[str] = []
            while len(chosen) < k:
                for j, sid in enumerate(pool):
                    if sid not in chosen:
                        chosen.append(pool.pop(j))
                        break
                else:
                    fresh = list(self._ids)
                    rng.shuffle(fresh)
                    pool.extend(fresh)
            return chosen

        cases: list[Case] = []
        seen: set[str] = set()
        for i, feats in enumerate(combos):
            k = 2 if feats["bundling"] == "bundled" else 1
            case_seed = _derive_int("case", seed, i) % (2**31)
            case = self.render_sets(draw(k), feats, case_seed)
            if case.id in seen:  # pragma: no cover - would need a sha256 prefix collision
                raise RuntimeError(f"duplicate case id {case.id}")
            seen.add(case.id)
            cases.append(case)
        return cases


# --- Metamorphic relation --------------------------------------------------------


def reword_ask(case: Case, seed: int) -> dict[str, Any]:
    """Return a new `input` dict with the same meaning as `case.input` but different wording.

    For cases built by this renderer, the ask is recomposed from the stored
    plan: atom phrases, actor, topics and asides are kept verbatim while the
    framing sentences, greetings, sign-offs, sentence order (the lead atom of
    each set stays first) and set order change, and a fixed synonym table is
    applied to the framing. Features (format, tone, actor presence) are kept.

    For other cases (no stored plan), the fixed synonym table is applied to the
    text, and a greeting is added if nothing changed.
    """
    original = str(case.input.get("ask", ""))
    source = case.source or {}
    has_plan = bool(source.get("sets")) and all(
        isinstance(a, dict) and {"set_id", "mode", "phrase", "form"} <= set(a) for a in source.get("atoms", [])
    )
    if has_plan and source.get("atoms"):
        feats = normalize_features({k: v for k, v in case.strata.items() if k in FEATURE_VALUES})
        for attempt in range(20):
            rng = random.Random(_derive_int("reword", seed, case.id, attempt))
            text = _compose(source, feats, rng, reorder=True, swap=True)
            if text != original:
                return {**case.input, "ask": text}
    text = _swap_synonyms(original)
    if text == original:
        text = "Hi team,\n\n" + original
    return {**case.input, "ask": text}
