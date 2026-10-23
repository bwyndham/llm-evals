from __future__ import annotations

import pytest

from llm_evals.store.sqlite import Store


@pytest.fixture
def store() -> Store:
    return Store(":memory:")
