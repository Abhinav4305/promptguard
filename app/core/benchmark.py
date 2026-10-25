"""
Versioned benchmarks: validation and identity of test cases.

A benchmark is (name, version) plus cases with stable keys. Once published, a case
definition must not change under the same (name, version, case_key): scores from
different dates would otherwise compare different questions. `case_content_hash`
makes that checkable, and the import endpoint refuses a changed case (HTTP 409);
changing a case means publishing a new benchmark version.
"""

import hashlib
import json
import re
from typing import Any, Sequence

from app.core.evaluators import EvaluatorConfigError, resolve_evaluators

CASE_KEY_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,99}$")
NAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,99}$")

TASK_TYPES = {"qa", "classification", "extraction", "structured_output", "instruction_following", "summarization"}
CATEGORIES = {
    "normal",             # ordinary, unambiguous requests
    "ambiguous",          # more than one defensible answer
    "edge",               # boundary values, unusual but valid input
    "long_input",         # long prompt/context
    "long_output",        # answer is expected to be long
    "empty_malformed",    # empty, garbled or contradictory input
    "structured_output",  # JSON / fixed-format answers
    "similarity_unreliable",  # right and wrong answers look alike to text similarity
}
DIFFICULTIES = {"easy", "medium", "hard"}
SUITES = {"pr", "full"}


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def case_content_hash(
    *, input_query: str, expected_output: str, eval_spec: Sequence[dict] | None,
    task_type: str | None, category: str | None, difficulty: str | None, suites: Sequence[str] | None,
) -> str:
    """sha256 over everything that defines what the case asks and how it is graded."""
    payload = {
        "input_query": input_query,
        "expected_output": expected_output,
        "eval_spec": list(eval_spec) if eval_spec else None,
        "task_type": task_type,
        "category": category,
        "difficulty": difficulty,
        "suites": sorted(suites) if suites else None,
    }
    return hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()


def validate_eval_spec(eval_spec: Sequence[dict[str, Any]] | None) -> None:
    """Instantiates every evaluator, so unknown names and bad configs fail at import time."""
    resolve_evaluators(eval_spec)


__all__ = [
    "CASE_KEY_RE", "NAME_RE", "TASK_TYPES", "CATEGORIES", "DIFFICULTIES", "SUITES",
    "case_content_hash", "validate_eval_spec", "EvaluatorConfigError",
]
