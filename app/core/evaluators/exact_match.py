import re
import string
from typing import Any

from app.core.evaluators.base import EvalInput, EvalResult, Evaluator, EvaluatorConfigError, EvaluatorKind
from app.core.evaluators.registry import register

_PUNCT = str.maketrans("", "", string.punctuation)


@register
class ExactMatch(Evaluator):
    """
    1.0 if the (normalised) output equals the reference or one of the accepted
    alternatives, else 0.0. Use it only where the exact output is the requirement
    (a label, a number, a fixed token), not for free text.

    config (all optional):
      case_sensitive       default true
      trim                 strip leading/trailing whitespace, default true
      collapse_whitespace  runs of whitespace -> one space, default false
      ignore_punctuation   drop ASCII punctuation, default false
      alternatives         extra accepted outputs, e.g. ["4", "four"]
    """

    name = "exact_match"
    version = "1"
    kind = EvaluatorKind.CORRECTNESS

    def validate_config(self, config: dict[str, Any]) -> None:
        allowed = {"case_sensitive", "trim", "collapse_whitespace", "ignore_punctuation", "alternatives"}
        unknown = set(config) - allowed
        if unknown:
            raise EvaluatorConfigError(f"exact_match: unknown config keys {sorted(unknown)}")
        alts = config.get("alternatives", [])
        if not isinstance(alts, list) or not all(isinstance(a, str) for a in alts):
            raise EvaluatorConfigError("exact_match: 'alternatives' must be a list of strings")

    def _norm(self, text: str) -> str:
        if self.config.get("trim", True):
            text = text.strip()
        if self.config.get("collapse_whitespace", False):
            text = re.sub(r"\s+", " ", text)
        if self.config.get("ignore_punctuation", False):
            text = text.translate(_PUNCT)
        if not self.config.get("case_sensitive", True):
            text = text.casefold()
        return text

    def evaluate(self, inp: EvalInput) -> EvalResult:
        prediction = self._norm(inp.prediction)
        candidates = [("reference", inp.reference), *[(f"alternative[{i}]", a) for i, a in enumerate(self.config.get("alternatives", []))]]
        for label, expected in candidates:
            if prediction == self._norm(expected):
                return EvalResult(1.0, {"matched": label})
        return EvalResult(0.0, {"matched": None, "normalized_prediction": prediction[:200]})
