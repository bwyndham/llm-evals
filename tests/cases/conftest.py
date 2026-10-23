from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SUITE_DIR = REPO / "suites" / "spec-writing"
ATOMS = SUITE_DIR / "data" / "atoms.jsonl"


@pytest.fixture(scope="session")
def suite_dir() -> Path:
    return SUITE_DIR


@pytest.fixture(scope="session")
def renderer():
    from llm_evals.cases.renderers.ask_renderer import AskRenderer

    return AskRenderer(ATOMS)
