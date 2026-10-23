"""Built-in checks.

Importing this package registers every check with `llm_evals.registry`: the
generic ones any suite can use (`field_matches`, `regex`, `keywords_present`,
`max_length`, and the `checklist` and `pairwise` judges), plus the checks
written for the spec-writing and research-agent pilots.
"""

from llm_evals.checks import format_rules, gherkin, invariance
from llm_evals.checks.gherkin import analyze_gherkin, scenario_count
from llm_evals.checks import research  # noqa: F401
from llm_evals.checks import faithfulness  # noqa: F401
from llm_evals.checks import checklist  # noqa: F401
from llm_evals.checks import pairwise  # noqa: F401
from llm_evals.checks import atom_keywords  # noqa: F401
from llm_evals.checks import bundles_split  # noqa: F401
from llm_evals.checks import gaps_flagged  # noqa: F401
from llm_evals.checks import generic  # noqa: F401

__all__ = ["analyze_gherkin", "format_rules", "gherkin", "invariance", "scenario_count"]
