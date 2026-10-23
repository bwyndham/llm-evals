"""An offline stand-in for the model, used with `--provider mock`.

It is not a model: it sorts messages by keyword and builds a reply from
templates. It exists so you can run the whole pipeline (cases, checks, gate,
report) without an API key. Its behaviour depends on the system prompt the way
a real model's would, so the example v2 prompt really does change the result:
v2 asks for a next step and forbids blame, and the stand-in follows that.
Delete this file and the `mock:` line once you run against the real model.
"""

from __future__ import annotations

import re

CATEGORIES = {
    "billing": ("charged", "refund", "invoice", "plan", "payment", "billing", "card"),
    "bug": ("crash", "doesn't", "does nothing", "stuck", "broken", "error", "sync", "export"),
    "account": ("password", "sign in", "log in", "access", "two-factor", "invited", "account"),
}
VOCABULARY = ("refund", "invoice", "plan", "payment", "export", "reports", "sync", "password", "access",
              "two-factor", "API", "thank")


def respond(*, system: str | None, user: str, rng) -> dict:
    found = re.search(r"<message>\s*(.*?)\s*</message>", user, re.DOTALL)
    message = found.group(1) if found else user
    text = message.lower()
    category = next((c for c, words in CATEGORIES.items() if any(w in text for w in words)), "other")
    if rng.random() < 0.08:  # a little sampling noise, like a real model
        category = "other"
    mentioned = [w for w in VOCABULARY if w.lower() in text]
    topic = " and ".join(mentioned[:2]) or "your message"
    reply = f"Thanks for getting in touch about the {topic}."
    strict = "next step" in (system or "").lower()
    if strict:
        reply += " Next step: our team will look into it and reply within one business day."
    elif rng.random() < 0.3:
        reply += " Please make sure you followed the setup guide before contacting us."
    else:
        reply += " We're looking into it."
    return {"category": category, "reply": reply}
