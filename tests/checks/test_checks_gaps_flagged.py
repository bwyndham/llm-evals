"""gaps_flagged: asks with a known gap must get an assumption or open question."""

from llm_evals.checks.gaps_flagged import gaps_flagged
from llm_evals.types import Case, Result


def _case(missing_actor="no", hinted=()):
    return Case(id="c", input={"ask": "..."}, provenance="constructed",
                strata={"missing_actor": missing_actor},
                source={"atoms": [], "hinted_atom_ids": list(hinted), "actor_stated": missing_actor == "no"})


def _result(assumptions=(), questions=()):
    return Result(run_id="r", case_id="c", version_id="v", sample_idx=0,
                  output={"title": "t", "description": "d", "acceptance_criteria": "",
                          "assumptions": list(assumptions), "open_questions": list(questions)})


def test_missing_actor_needs_a_flag():
    assert not gaps_flagged().score(_case("yes"), _result(), None).passed
    assert gaps_flagged().score(_case("yes"), _result(assumptions=["User is an admin"]), None).passed


def test_hinted_requirement_needs_a_flag():
    assert not gaps_flagged().score(_case(hinted=["s/a3"]), _result(), None).passed
    assert gaps_flagged().score(_case(hinted=["s/a3"]), _result(questions=["PDF or CSV?"]), None).passed


def test_blank_strings_do_not_count():
    assert not gaps_flagged().score(_case("yes"), _result(assumptions=["  "]), None).passed


def test_no_gap_is_not_applicable():
    s = gaps_flagged().score(_case(), _result(), None)
    assert s.passed and s.reason.startswith("not applicable")
