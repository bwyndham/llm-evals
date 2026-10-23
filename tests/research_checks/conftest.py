from __future__ import annotations

import pytest

from llm_evals.checks import research


@pytest.fixture(autouse=True)
def local_matcher(monkeypatch):
    """Test this module's matcher, not whatever the corpus helper does today.

    `test_prefers_corpus_helper` restores the real lookup explicitly.
    """
    monkeypatch.setattr(research, "_corpus_matcher", lambda: None)
