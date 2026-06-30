from datetime import datetime
from pydantic import BaseModel, Field
from app.db.models.evaluation_run import RunStatus

class EvaluationRunRequest(BaseModel):
    prompt_id: int
    dataset_ids: list[int]
    is_baseline: bool = False

class EvaluationResultResponse(BaseModel):
    id: int
    run_id: int
    dataset_id: int
    latency_seconds: float | None
    token_cost: float | None
    similarity_score: float | None
    actual_output: str | None
    created_at: datetime
    model_config = {"from_attributes": True}

class EvaluationRunResponse(BaseModel):
    id: int
    prompt_id: int
    status: RunStatus
    is_baseline: bool
    created_at: datetime
    completed_at: datetime | None
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
    baseline_similarity: float | None
    candidate_similarity: float | None
    similarity_delta: float | None
    similarity_delta_pct: float | None
    baseline_latency_s: float | None
    candidate_latency_s: float | None
    latency_delta_s: float | None

class ComparisonResponse(BaseModel):
    baseline_run_id: int
    candidate_run_id: int
    passed: bool
    summary: str
    similarity: MetricDeltaResponse
    latency: MetricDeltaResponse
    cost: MetricDeltaResponse
    per_dataset: list[PerDatasetRow] = []