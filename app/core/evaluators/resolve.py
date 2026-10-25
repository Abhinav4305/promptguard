"""
Turning a test case's evaluator spec into evaluator instances, and deciding which
scores are allowed to block a release.

A case's `eval_spec` is a list like
    [{"name": "contains", "config": {"all": ["paris"]}}, {"name": "text_similarity"}]
A case with no spec uses the original behaviour: text_similarity alone.

Gating rule (recorded on every stored score, so history stays interpretable):
  * a CORRECTNESS evaluator gates by default
  * a SIMILARITY evaluator gates only if the case has no correctness evaluator;
    next to an objective check it is reported but advisory
  * an entry can override with "gate": true/false
"""

from dataclasses import dataclass
from typing import Any, Sequence

from app.core.evaluators import registry
from app.core.evaluators.base import Evaluator, EvaluatorConfigError, EvaluatorKind

DEFAULT_EVAL_SPEC: list[dict[str, Any]] = [{"name": "text_similarity"}]


@dataclass(frozen=True)
class ResolvedEvaluator:
    evaluator: Evaluator
    gating: bool


def resolve_evaluators(eval_spec: Sequence[dict[str, Any]] | None) -> list[ResolvedEvaluator]:
    """Raises EvaluatorConfigError for unknown names, bad configs or duplicate evaluators."""
    spec = list(eval_spec) if eval_spec else DEFAULT_EVAL_SPEC
    built: list[tuple[Evaluator, bool | None]] = []
    seen: set[str] = set()
    for entry in spec:
        if not isinstance(entry, dict) or "name" not in entry:
            raise EvaluatorConfigError(f"evaluator entry must be an object with a 'name': {entry!r}")
        unknown = set(entry) - {"name", "config", "gate"}
        if unknown:
            raise EvaluatorConfigError(f"evaluator entry {entry['name']!r}: unknown keys {sorted(unknown)}")
        if entry["name"] in seen:
            raise EvaluatorConfigError(f"evaluator {entry['name']!r} is listed twice for one case")
        seen.add(entry["name"])
        built.append((registry.create_evaluator(entry["name"], entry.get("config")), entry.get("gate")))

    has_correctness = any(ev.kind == EvaluatorKind.CORRECTNESS for ev, _ in built)
    resolved = []
    for ev, gate in built:
        if gate is None:
            gate = ev.kind == EvaluatorKind.CORRECTNESS or not has_correctness
        resolved.append(ResolvedEvaluator(ev, bool(gate)))
    return resolved
