import pytest

from llm_evals.cases.corpus import answer_matches, normalize_answer, text_mentions


@pytest.mark.parametrize(
    ("raw", "norm"),
    [
        ("48,200", "48200"),
        ("1,840 km²", "1840 km2"),
        ("1,840 square kilometres", "1840 km2"),
        ("84km", "84 km"),
        ("84 kilometers", "84 km"),
        ("1,486 metres", "1486 m"),
        ("12.50", "12.5"),
        ("1887.0", "1887"),
        ("The Greywater Bridge", "greywater bridge"),
        ("Mayor Ilse Varga", "ilse varga"),
        ("Ilvane Solar Co-op", "ilvane solar co op"),
        ("  Port   Ilvane. ", "port ilvane"),
        ("12 per cent", "12 %"),
        (None, ""),
        (1887, "1887"),
    ],
)
def test_normalize_answer(raw, norm) -> None:
    assert normalize_answer(raw) == norm


@pytest.mark.parametrize(
    ("output", "expected", "ok"),
    [
        ("1887", "1887", True),
        ("The town was founded in 1847.", "1847", True),
        ("12,500 people", "12500", True),
        ("12500.0", "12,500", True),
        ("about 84 kilometres", "84 km", True),
        ("84", "84 km", True),
        ("84 m", "84 km", False),
        ("1,840 square kilometres", "1,840 km²", True),
        ("18870", "1887", False),
        ("1886", "1887", False),
        ("the Kestrin Lamp Company", "Kestrin Lamp Company", True),
        ("Ilse Varga", "Ilse Varga", True),
        ("The mayor is Ilse Varga.", "Ilse Varga", True),
        ("Varga", "Ilse Varga", False),  # aliases come from the expected dict
        ("", "1887", False),
        (None, "1887", False),
        ("1887", None, False),
    ],
)
def test_answer_matches_strings(output, expected, ok) -> None:
    assert answer_matches(output, expected) is ok


def test_answer_matches_uses_expected_dict() -> None:
    exp = {"behavior": "answer", "answer": "Ilse Varga", "aliases": ["Varga"], "original_answer": None,
           "distractors": []}
    assert answer_matches("Varga", exp)
    assert answer_matches("Mayor Varga", exp)
    assert not answer_matches("Marisol Achterberg", exp)
    assert not answer_matches("anything", {"behavior": "abstain", "answer": None, "aliases": []})
    assert answer_matches("Kestrin Lamp", "Kestrin Lamp Company", aliases=["Kestrin Lamp"])


def test_distractors_block_look_alikes_and_comparison_losers() -> None:
    exp = {"answer": "Drennick", "aliases": [], "distractors": ["New Drennick"]}
    assert answer_matches("Drennick", exp)
    assert answer_matches("It is headquartered in Drennick.", exp)
    assert not answer_matches("New Drennick", exp)
    cmp = {"answer": "Velmouth", "aliases": [], "distractors": ["Thornquay"]}
    assert answer_matches("Velmouth has the larger population.", cmp)
    assert not answer_matches("Thornquay is larger than Velmouth", cmp)
    # a distractor contained in the answer itself is ignored
    assert answer_matches("Kestrin Lamp Company", "Kestrin Lamp Company", distractors=["Kestrin"])


def test_text_mentions() -> None:
    text = "Drennick had a population of 12,450. Harrowmark covers 1,840 km²."
    assert text_mentions(text, "12,450")
    assert text_mentions(text, "x", ["12450"])
    assert text_mentions(text, "1840 square kilometres")
    assert not text_mentions(text, "1245")
    assert not text_mentions(text, "450")
    assert text_mentions("New Drennick", "Drennick")
