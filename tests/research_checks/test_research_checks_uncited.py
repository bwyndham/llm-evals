from __future__ import annotations

import pytest
from research_checks_helpers import FakeContext, make_case, make_output, make_result

from llm_evals.checks.research import uncited_claims


def run(output, **params):
    return uncited_claims(**params).score(make_case(), make_result(output), FakeContext())


def claims(cited: int, uncited: int):
    return [{"text": f"cited {i}", "citations": ["d1"]} for i in range(cited)] + [
        {"text": f"uncited {i}", "citations": []} for i in range(uncited)
    ]


def test_defaults():
    check = uncited_claims()
    assert (check.name, check.kind, check.critical, check.max_share) == ("uncited_claims", "structural", False, 0.1)


def test_all_cited_passes():
    score = run(make_output(claims=claims(3, 0)))
    assert score.passed and score.value == 1.0 and score.sub_scores == []


def test_share_at_threshold_passes():
    score = run(make_output(claims=claims(9, 1)))
    assert score.passed and score.value == pytest.approx(0.9)
    assert score.sub_scores == [{"claim": 9, "text": "uncited 0"}]


def test_share_over_threshold_fails():
    score = run(make_output(claims=claims(3, 1)))
    assert not score.passed
    assert score.reason == "1 of 4 claim(s) uncited (25%, max 10%)"


def test_custom_max_share():
    assert run(make_output(claims=claims(3, 1)), max_share=0.25).passed
    assert not run(make_output(claims=claims(1, 0) + [{"text": "x"}]), max_share=0.0).passed


def test_missing_or_blank_citations_count_as_uncited():
    output = make_output(claims=[{"text": "a"}, {"text": "b", "citations": [" "]}, {"text": "c", "citations": ["d1"]}])
    score = run(output, max_share=0.5)
    assert not score.passed and [s["claim"] for s in score.sub_scores] == [0, 1]


def test_answered_without_claims_is_fully_uncited():
    score = run(make_output(claims=[]))
    assert not score.passed and score.value == 0.0 and "no claims" in score.reason


def test_not_found_without_claims_passes():
    assert run(make_output(status="not_found", answer="", claims=[])).passed


@pytest.mark.parametrize("bad", [-0.1, 1.5, "0.1", True, None])
def test_bad_max_share_rejected(bad):
    with pytest.raises(TypeError, match="max_share"):
        uncited_claims(max_share=bad)
