"""
Evaluator interface.

An evaluator turns (model output, reference, case metadata) into ONE score for ONE
test case. It knows nothing about Celery, the database or run comparison; the run
executor calls it and stores the outcome, and the comparator aggregates the stored
scores. New evaluators are added by subclassing Evaluator and registering them
(see registry.py): neither the executor nor the comparator needs to change.

Conventions every evaluator must follow (enforced at registration):
  * score is a finite float in value_range = [0.0, 1.0]
  * direction is "higher_is_better" (the comparator only supports that today)
  * aggregation across cases is the arithmetic "mean", and it is only defined when
    EVERY case has a valid score (see aggregate): missing scores are never skipped.

`kind` separates what a score means:
  * CORRECTNESS - an objective, task-specific check (exact match, schema validity,
                  required facts, structured equality). Can gate a release.
  * SIMILARITY  - surface text similarity (ROUGE-L / Levenshtein). An indicator, not
                  a measure of correctness; it gates only when a case has no
                  correctness evaluator (see resolve.py).

A future rubric-based LLM judge fits this interface: kind CORRECTNESS, config carrying
the rubric, judge model and prompt version, details carrying the judge's rationale.
It needs no schema or comparator change (see docs/evaluators.md for the calibration
plan).
"""

import hashlib
import json
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, ClassVar, Mapping, Sequence

SUPPORTED_DIRECTIONS = {"higher_is_better"}
SUPPORTED_AGGREGATIONS = {"mean"}
SCORE_MIN, SCORE_MAX = 0.0, 1.0


class EvaluatorKind(str, Enum):
    CORRECTNESS = "correctness"
    SIMILARITY = "similarity"


class EvaluatorConfigError(ValueError):
    """The evaluator's config is invalid (a benchmark/authoring error, not a model failure)."""


class EvaluatorError(Exception):
    """The evaluator could not produce a score (e.g. the reference itself is unusable)."""


class MissingScoreError(ValueError):
    """Aggregation was asked to average over a missing or invalid score."""


@dataclass(frozen=True)
class EvalInput:
    prediction: str
    reference: str
    case: Mapping[str, Any] = field(default_factory=dict)   # task_type, category, ... (for future judges)


@dataclass
class EvalResult:
    score: float
    details: dict[str, Any] = field(default_factory=dict)   # raw sub-scores / diagnostics, stored as-is


class OutcomeStatus(str, Enum):
    OK = "OK"
    ERROR = "ERROR"


@dataclass
class EvaluatorOutcome:
    """What the executor stores: a valid score, or an explicit error. Never a silent gap."""
    status: OutcomeStatus
    score: float | None
    details: dict[str, Any]
    error_message: str | None = None


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def config_hash(config: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(dict(config)).encode("utf-8")).hexdigest()[:16]


class Evaluator(ABC):
    name: ClassVar[str]
    version: ClassVar[str]
    kind: ClassVar[EvaluatorKind]
    direction: ClassVar[str] = "higher_is_better"
    aggregation: ClassVar[str] = "mean"
    value_range: ClassVar[tuple[float, float]] = (SCORE_MIN, SCORE_MAX)

    def __init__(self, config: Mapping[str, Any] | None = None):
        self.config: dict[str, Any] = dict(config or {})
        self.validate_config(self.config)

    # -- to implement -------------------------------------------------------------
    def validate_config(self, config: dict[str, Any]) -> None:
        """Raise EvaluatorConfigError for an invalid config. Default: accept anything."""

    @abstractmethod
    def evaluate(self, inp: EvalInput) -> EvalResult:
        """Return a score in value_range. Raise EvaluatorError if no score can be produced."""

    # -- provided -------------------------------------------------------------------
    @property
    def config_hash(self) -> str:
        return config_hash(self.config)

    def descriptor(self) -> dict[str, Any]:
        """Everything needed to interpret a stored score later."""
        return {
            "name": self.name,
            "version": self.version,
            "kind": self.kind.value,
            "direction": self.direction,
            "aggregation": self.aggregation,
            "value_range": list(self.value_range),
            "config": self.config,
            "config_hash": self.config_hash,
        }


def check_evaluator_class(cls: type[Evaluator]) -> None:
    """Registration-time contract check, so inconsistent evaluators never reach a run."""
    for attr in ("name", "version", "kind"):
        if not hasattr(cls, attr):
            raise TypeError(f"{cls.__name__} must define class attribute '{attr}'")
    if cls.direction not in SUPPORTED_DIRECTIONS:
        raise TypeError(f"{cls.name}: direction {cls.direction!r} is not supported by the comparator")
    if cls.aggregation not in SUPPORTED_AGGREGATIONS:
        raise TypeError(f"{cls.name}: aggregation {cls.aggregation!r} is not supported")
    if tuple(cls.value_range) != (SCORE_MIN, SCORE_MAX):
        raise TypeError(f"{cls.name}: value_range must be {(SCORE_MIN, SCORE_MAX)}")


def is_valid_score(score: Any) -> bool:
    return (
        isinstance(score, (int, float))
        and not isinstance(score, bool)
        and math.isfinite(score)
        and SCORE_MIN <= score <= SCORE_MAX
    )


def run_evaluator(evaluator: Evaluator, inp: EvalInput) -> EvaluatorOutcome:
    """Run one evaluator and normalise every failure mode into an explicit ERROR outcome."""
    try:
        result = evaluator.evaluate(inp)
    except Exception as exc:  # noqa: BLE001 - an evaluator bug must be recorded, never raised into the run
        return EvaluatorOutcome(OutcomeStatus.ERROR, None, {}, f"{type(exc).__name__}: {exc}"[:1000])
    if not is_valid_score(result.score):
        return EvaluatorOutcome(
            OutcomeStatus.ERROR, None, dict(result.details),
            f"invalid score {result.score!r}: expected a finite number in [{SCORE_MIN}, {SCORE_MAX}]",
        )
    return EvaluatorOutcome(OutcomeStatus.OK, float(result.score), dict(result.details))


def aggregate(scores: Sequence[float | None]) -> float:
    """
    Mean over cases. Defined only if every case has a valid score: a missing, NaN or
    out-of-range score raises instead of being skipped, because averaging "whatever
    values happen to exist" makes incomplete runs look healthy.
    """
    if not scores:
        raise MissingScoreError("cannot aggregate zero scores")
    bad = [i for i, s in enumerate(scores) if not is_valid_score(s)]
    if bad:
        raise MissingScoreError(f"{len(bad)} of {len(scores)} scores are missing or invalid (positions {bad[:10]})")
    return sum(scores) / len(scores)  # type: ignore[arg-type]
