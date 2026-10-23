"""Judge calibration: human labels vs judge verdicts per gating unit (CAL-1, CAL-2, CAL-4)."""

from llm_evals.calibration.calibrate import calibrate, human_labels, judge_verdicts, unit_triples
from llm_evals.calibration.metrics import AgreementMetrics, agreement_metrics, from_confusion
from llm_evals.calibration.sampling import LabelSample, candidate_samples, choose_samples
from llm_evals.store.sqlite import CalibrationRecord, LabelRecord

__all__ = [
    "AgreementMetrics",
    "CalibrationRecord",
    "LabelRecord",
    "LabelSample",
    "agreement_metrics",
    "calibrate",
    "candidate_samples",
    "choose_samples",
    "from_confusion",
    "human_labels",
    "judge_verdicts",
    "unit_triples",
]
