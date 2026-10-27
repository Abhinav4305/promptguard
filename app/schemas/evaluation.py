from datetime import datetime
from pydantic import BaseModel, Field
from app.core.comparator import ComparisonStatus
from app.db.models.evaluation_result import ErrorCategory, ResultStatus
from app.db.models.evaluation_run import RunStatus
from app.db.models.evaluation_score import ScoreStatus

class EvaluationRunRequest(BaseModel):
    prompt_id: int
    # Test cases to run. An empty list means "every dataset that exists right now";
    # the list is resolved when the run is created and then fixed for the run.
    dataset_ids: list[int]
    is_baseline: bool = False

class EvaluationScoreResponse(BaseModel):
    evaluator_name: str
    evaluator_version: str
    kind: str
    config: dict | None = None
    config_hash: str
    gating: bool
    status: ScoreStatus
    score: float | None = None
    details: dict | None = None
    error_message: str | None = None
    model_config = {"from_attributes": True}

class EvaluationResultResponse(BaseModel):
    id: int
    run_id: int
    dataset_id: int
    # -- execution (added): a FAILED case carries a category and a diagnostic message
    status: ResultStatus
    error_category: ErrorCategory | None = None
    error_message: str | None = None
    attempts: int = 0
    # -- measurements and model output (unchanged; NULL when the case did not succeed)
    latency_seconds: float | None
    token_cost: float | None
    similarity_score: float | None      # compatibility copy of the text_similarity score
    actual_output: str | None
    created_at: datetime
    # One entry per evaluator: name, version, config, gating flag, score or ERROR.
    scores: list[EvaluationScoreResponse] = []
    model_config = {"from_attributes": True}

class EvaluationRunResponse(BaseModel):
    id: int
    prompt_id: int
    status: RunStatus
    is_baseline: bool
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None
    # Case counters. status == COMPLETED implies failed_cases == 0 and
    # succeeded_cases == expected_cases.
    expected_cases: int = 0
    completed_cases: int = 0
    succeeded_cases: int = 0
    failed_cases: int = 0
    error_message: str | None = None
    results: list[EvaluationResultResponse] = Field(
        default_factory=list, validation_alias="evaluation_results"
    )
    model_config = {"from_attributes": True, "populate_by_name": True}

class MetricDeltaResponse(BaseModel):
    baseline_avg: float | None
    candidate_avg: float | None
    delta_absolute: float | None
    delta_relative: float | None
    regression_detected: bool
    threshold_used: float

class PerDatasetRow(BaseModel):
    dataset_id: int
    baseline_quality: float | None = None      # mean of the case's gating scores
    candidate_quality: float | None = None
    quality_delta: float | None = None
    quality_delta_pct: float | None = None
    baseline_similarity: float | None
    candidate_similarity: float | None
    similarity_delta: float | None
    similarity_delta_pct: float | None
    baseline_latency_s: float | None
    candidate_latency_s: float | None
    latency_delta_s: float | None
    # evaluator name -> {baseline, candidate, delta, gating, kind, version}
    scores: dict[str, dict] = {}

class EvaluatorDeltaResponse(BaseModel):
    name: str
    version: str
    kind: str
    gating: bool
    n_cases: int
    baseline_avg: float
    candidate_avg: float
    delta_absolute: float
    delta_relative: float | None
    threshold_used: float
    regression_detected: bool
    advisory_regression: bool

class RegressedCaseResponse(BaseModel):
    dataset_id: int
    evaluator: str
    baseline_score: float
    candidate_score: float

class ComparisonIssueResponse(BaseModel):
    code: str
    severity: ComparisonStatus
    run: str                      # "baseline" | "candidate" | "both"
    message: str
    dataset_ids: list[int] = []

class ComparisonResponse(BaseModel):
    baseline_run_id: int
    candidate_run_id: int
    # OK            -> runs are comparable; `passed` is the quality verdict
    # INCOMPLETE    -> a run has failed/unfinished/unscored cases
    # INCOMPARABLE  -> runs do not cover the same cases
    # For the last two, passed is always False and no metric verdict is produced.
    comparison_status: ComparisonStatus = ComparisonStatus.OK
    policy_version: str = ""
    passed: bool
    summary: str
    issues: list[ComparisonIssueResponse] = []
    # quality = mean of the gating scores per case, averaged over cases; regression_detected is
    # True when ANY gating check dropped past its threshold (see `evaluators`), even if the
    # overall mean moved less than the threshold.
    quality: MetricDeltaResponse | None = None
    similarity: MetricDeltaResponse | None = None   # legacy: text_similarity only
    latency: MetricDeltaResponse | None = None
    cost: MetricDeltaResponse | None = None
    per_dataset: list[PerDatasetRow] = []
    evaluators: list[EvaluatorDeltaResponse] = []
    regressed_cases: list[RegressedCaseResponse] = []
