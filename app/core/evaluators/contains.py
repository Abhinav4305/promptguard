import re
from typing import Any

from app.core.evaluators.base import EvalInput, EvalResult, Evaluator, EvaluatorConfigError, EvaluatorKind
from app.core.evaluators.registry import register


@register
class Contains(Evaluator):
    """
    Checks facts rather than wording: 1.0 if the output contains every `all` term,
    at least one `any` term, and none of the `none` terms; otherwise 0.0.

    This is the usual way to grade free-text answers where many phrasings are
    correct: "Paris is the capital" and "The capital of France is Paris." both pass
    `all: ["paris"]`, while "The capital is Lyon." fails it.

    config (at least one of all / any / none is required):
      all           list[str] required terms
      any           list[str] at least one must appear
      none          list[str] forbidden terms
      case_sensitive  default false
      whole_word      default true: "4" does not match inside "14"
    """

    name = "contains"
    version = "1"
    kind = EvaluatorKind.CORRECTNESS

    def validate_config(self, config: dict[str, Any]) -> None:
        allowed = {"all", "any", "none", "case_sensitive", "whole_word"}
        unknown = set(config) - allowed
        if unknown:
            raise EvaluatorConfigError(f"contains: unknown config keys {sorted(unknown)}")
        if not any(config.get(k) for k in ("all", "any", "none")):
            raise EvaluatorConfigError("contains: give at least one of 'all', 'any', 'none'")
        for key in ("all", "any", "none"):
            terms = config.get(key, [])
            if not isinstance(terms, list) or not all(isinstance(t, str) and t for t in terms):
                raise EvaluatorConfigError(f"contains: '{key}' must be a list of non-empty strings")

    def _found(self, term: str, text: str) -> bool:
        flags = 0 if self.config.get("case_sensitive", False) else re.IGNORECASE
        body = re.escape(term)
        pattern = rf"(?<!\w){body}(?!\w)" if self.config.get("whole_word", True) else body
        return re.search(pattern, text, flags) is not None

    def evaluate(self, inp: EvalInput) -> EvalResult:
        text = inp.prediction
        missing = [t for t in self.config.get("all", []) if not self._found(t, text)]
        any_terms = self.config.get("any", [])
        any_ok = (not any_terms) or any(self._found(t, text) for t in any_terms)
        forbidden = [t for t in self.config.get("none", []) if self._found(t, text)]
        ok = not missing and any_ok and not forbidden
        return EvalResult(
            1.0 if ok else 0.0,
            {"missing_required": missing, "any_matched": any_ok, "forbidden_found": forbidden},
        )
