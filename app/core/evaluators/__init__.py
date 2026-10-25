"""Importing this package registers every built-in evaluator."""

from app.core.evaluators import contains, exact_match, json_eval, text_similarity  # noqa: F401
from app.core.evaluators.base import (  # noqa: F401
    EvalInput, EvalResult, Evaluator, EvaluatorConfigError, EvaluatorError, EvaluatorKind,
    EvaluatorOutcome, MissingScoreError, OutcomeStatus, aggregate, run_evaluator,
)
from app.core.evaluators.registry import available, create_evaluator, register  # noqa: F401
from app.core.evaluators.resolve import DEFAULT_EVAL_SPEC, ResolvedEvaluator, resolve_evaluators  # noqa: F401
