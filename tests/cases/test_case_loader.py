import json
from pathlib import Path

import pytest

from llm_evals.cases import CaseLoadError, case_set_hash, load_cases, write_cases
from llm_evals.types import Case


def _case(i: str, **kw) -> Case:
    return Case(id=i, input={"ask": f"ask {i}"}, provenance="constructed", strata={"x": "1"}, **kw)


def test_round_trip(tmp_path: Path) -> None:
    cases = [_case("b", source={"atoms": [{"id": "s/a1"}]}), _case("a"), _case("c", tags=["t"])]
    path = write_cases(tmp_path / "sub" / "cases.jsonl", cases)
    loaded = load_cases(path)
    assert [c.id for c in loaded] == ["a", "b", "c"]  # written sorted by id
    assert {c.id: c for c in loaded} == {c.id: c for c in cases}
    lines = path.read_text().splitlines()
    assert len(lines) == 3
    for line in lines:  # canonical: sorted keys, compact
        data = json.loads(line)
        assert line == json.dumps(data, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    # writing the loaded cases again gives identical bytes
    again = write_cases(tmp_path / "again.jsonl", reversed(loaded))
    assert again.read_bytes() == path.read_bytes()


def test_duplicate_ids_rejected(tmp_path: Path) -> None:
    path = tmp_path / "dup.jsonl"
    rows = [_case("a"), _case("b"), _case("a")]
    path.write_text("".join(r.model_dump_json() + "\n" for r in rows))
    with pytest.raises(CaseLoadError) as err:
        load_cases(path)
    assert err.value.line == 3
    assert "duplicate" in str(err.value) and "line 1" in str(err.value)
    with pytest.raises(ValueError):
        write_cases(tmp_path / "out.jsonl", rows)


@pytest.mark.parametrize(
    "bad",
    ["{not json", json.dumps({"id": "x", "input": {}}), json.dumps(["a list"])],
    ids=["bad-json", "missing-provenance", "not-object"],
)
def test_bad_line_reports_line_number(tmp_path: Path, bad: str) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text(_case("a").model_dump_json() + "\n\n" + bad + "\n")
    with pytest.raises(CaseLoadError) as err:
        load_cases(path)
    assert err.value.line == 3
    assert f"{path}:3" in str(err.value)


def test_blank_lines_skipped(tmp_path: Path) -> None:
    path = tmp_path / "c.jsonl"
    path.write_text("\n" + _case("a").model_dump_json() + "\n\n")
    assert [c.id for c in load_cases(path)] == ["a"]


def test_missing_file(tmp_path: Path) -> None:
    with pytest.raises(CaseLoadError):
        load_cases(tmp_path / "nope.jsonl")


def test_hash_order_independent_and_content_sensitive() -> None:
    cases = [_case("a"), _case("b"), _case("c")]
    h = case_set_hash(cases)
    assert len(h) == 64 and int(h, 16) >= 0
    assert case_set_hash(list(reversed(cases))) == h
    assert case_set_hash([cases[1], cases[2], cases[0]]) == h
    changed = [_case("a"), _case("b"), _case("c", tags=["new"])]
    assert case_set_hash(changed) != h
    assert case_set_hash(cases[:2]) != h
