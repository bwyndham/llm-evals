from __future__ import annotations

import pytest
from checks_helpers import FakeContext, make_result

from llm_evals.types import Case, ErrorKind, Result


@pytest.fixture
def case() -> Case:
    return Case(
        id="c1", input={"ask": "can admins get the usage thing as csv? last 90 days pls"}, provenance="constructed"
    )


@pytest.fixture
def ctx() -> FakeContext:
    return FakeContext()


@pytest.fixture
def error_results() -> list[Result]:
    return [
        make_result(None, error_kind=ErrorKind.SYSTEM, error="invalid JSON"),
        make_result(None, error_kind=ErrorKind.INFRA, error="timeout"),
        make_result(None),
    ]
