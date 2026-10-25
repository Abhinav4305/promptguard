"""Structured-output evaluators: JSON schema validity and JSON value equality."""

import json
import re
from typing import Any

import jsonschema

from app.core.evaluators.base import (
    EvalInput, EvalResult, Evaluator, EvaluatorConfigError, EvaluatorError, EvaluatorKind,
)
from app.core.evaluators.registry import register

_FENCE = re.compile(r"^\s*```(?:json)?\s*\n(.*?)\n?```\s*$", re.DOTALL | re.IGNORECASE)


def parse_json_output(text: str, lenient: bool = True) -> Any:
    """
    Parse model output as JSON. lenient=True accepts exactly one wrapper models add
    constantly, a ```json fenced block; nothing else is repaired. Raises ValueError.
    """
    candidate = text.strip()
    if lenient:
        fenced = _FENCE.match(candidate)
        if fenced:
            candidate = fenced.group(1).strip()
    return json.loads(candidate)


@register
class JsonSchema(Evaluator):
    """
    1.0 if the output parses as JSON AND validates against `schema`, else 0.0.
    Checks the SHAPE of a structured answer, not that its values are right: pair it
    with json_equal (or contains) when values matter.

    config:
      schema   JSON Schema (draft 2020-12), required
      lenient  accept a ```json fenced block around the JSON, default true
    """

    name = "json_schema"
    version = "1"
    kind = EvaluatorKind.CORRECTNESS

    def validate_config(self, config: dict[str, Any]) -> None:
        unknown = set(config) - {"schema", "lenient"}
        if unknown:
            raise EvaluatorConfigError(f"json_schema: unknown config keys {sorted(unknown)}")
        schema = config.get("schema")
        if not isinstance(schema, (dict, bool)):
            raise EvaluatorConfigError("json_schema: 'schema' (a JSON Schema object) is required")
        try:
            jsonschema.Draft202012Validator.check_schema(schema)
        except jsonschema.SchemaError as exc:
            raise EvaluatorConfigError(f"json_schema: invalid schema: {exc.message}") from exc

    def evaluate(self, inp: EvalInput) -> EvalResult:
        try:
            value = parse_json_output(inp.prediction, self.config.get("lenient", True))
        except ValueError as exc:
            return EvalResult(0.0, {"stage": "parse", "error": str(exc)[:200]})
        validator = jsonschema.Draft202012Validator(self.config["schema"])
        errors = sorted(validator.iter_errors(value), key=lambda e: list(e.absolute_path))
        if errors:
            return EvalResult(0.0, {
                "stage": "validate",
                "errors": [f"{'/'.join(map(str, e.absolute_path)) or '<root>'}: {e.message}"[:200] for e in errors[:5]],
            })
        return EvalResult(1.0, {"stage": "ok"})


@register
class JsonEqual(Evaluator):
    """
    Compares the parsed output with the parsed reference.

    For an object reference the score is the FRACTION of reference keys whose values
    match (partial credit, so one wrong field is visible instead of a flat 0). Output
    that is not valid JSON scores 0.0. For a non-object reference the score is 1.0 on
    equality, else 0.0. Key order and whitespace never matter.

    config (all optional):
      fields            only compare these top-level keys
      ignore_keys       skip these top-level keys
      number_tolerance  absolute tolerance for numbers, default 0
      strict_keys       also require that the output has no extra top-level keys, default false
      lenient           accept a ```json fenced block, default true
    """

    name = "json_equal"
    version = "1"
    kind = EvaluatorKind.CORRECTNESS

    def validate_config(self, config: dict[str, Any]) -> None:
        allowed = {"fields", "ignore_keys", "number_tolerance", "strict_keys", "lenient"}
        unknown = set(config) - allowed
        if unknown:
            raise EvaluatorConfigError(f"json_equal: unknown config keys {sorted(unknown)}")
        for key in ("fields", "ignore_keys"):
            if key in config and not (isinstance(config[key], list) and all(isinstance(k, str) for k in config[key])):
                raise EvaluatorConfigError(f"json_equal: '{key}' must be a list of strings")
        tol = config.get("number_tolerance", 0)
        if isinstance(tol, bool) or not isinstance(tol, (int, float)) or tol < 0:
            raise EvaluatorConfigError("json_equal: 'number_tolerance' must be a number >= 0")

    def _equal(self, a: Any, b: Any) -> bool:
        tol = self.config.get("number_tolerance", 0)
        if isinstance(a, bool) or isinstance(b, bool):
            return a is b
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            return abs(a - b) <= tol
        if isinstance(a, dict) and isinstance(b, dict):
            return a.keys() == b.keys() and all(self._equal(a[k], b[k]) for k in a)
        if isinstance(a, list) and isinstance(b, list):
            return len(a) == len(b) and all(self._equal(x, y) for x, y in zip(a, b))
        return a == b

    def evaluate(self, inp: EvalInput) -> EvalResult:
        lenient = self.config.get("lenient", True)
        try:
            expected = parse_json_output(inp.reference, lenient)
        except ValueError as exc:
            raise EvaluatorError(f"reference is not valid JSON: {exc}") from exc
        try:
            actual = parse_json_output(inp.prediction, lenient)
        except ValueError as exc:
            return EvalResult(0.0, {"stage": "parse", "error": str(exc)[:200]})

        if not isinstance(expected, dict):
            equal = self._equal(actual, expected)
            return EvalResult(1.0 if equal else 0.0, {"stage": "compare", "equal": equal})

        keys = self.config.get("fields") or list(expected)
        keys = [k for k in keys if k not in set(self.config.get("ignore_keys", []))]
        if not keys:
            raise EvaluatorError("json_equal: no keys left to compare after fields/ignore_keys")
        if not isinstance(actual, dict):
            return EvalResult(0.0, {"stage": "compare", "error": "output is not a JSON object", "wrong_keys": keys})

        wrong = [k for k in keys if k not in actual or not self._equal(actual[k], expected.get(k))]
        extra = sorted(set(actual) - set(expected)) if self.config.get("strict_keys", False) else []
        score = (len(keys) - len(wrong)) / len(keys)
        if extra:
            score = 0.0
        return EvalResult(round(score, 4), {"stage": "compare", "wrong_keys": wrong, "extra_keys": extra})
