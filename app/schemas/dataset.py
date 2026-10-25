from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from app.core import benchmark as bm


def _check_spec(spec):
    if spec is None:
        return spec
    try:
        bm.validate_eval_spec(spec)
    except bm.EvaluatorConfigError as exc:
        raise ValueError(str(exc)) from exc
    return spec


class DatasetCreate(BaseModel):
    input_query: str
    expected_output: str
    # Optional: how to score this case. Omitted = text_similarity only (original behaviour).
    eval_spec: list[dict[str, Any]] | None = None

    _spec = field_validator("eval_spec")(_check_spec)


class DatasetResponse(BaseModel):
    id: int
    input_query: str
    expected_output: str
    created_at: datetime
    benchmark: str | None = None
    benchmark_version: str | None = None
    case_key: str | None = None
    task_type: str | None = None
    category: str | None = None
    difficulty: str | None = None
    suites: list[str] | None = None
    eval_spec: list[dict[str, Any]] | None = None
    content_hash: str | None = None
    model_config = {"from_attributes": True}


class BenchmarkCase(BaseModel):
    case_key: str
    input_query: str = Field(min_length=1)
    expected_output: str
    task_type: str
    category: str
    difficulty: str
    suites: list[str]
    # Benchmark cases must say how they are graded; there is no implicit default.
    eval_spec: list[dict[str, Any]] = Field(min_length=1)

    @field_validator("case_key")
    @classmethod
    def _key(cls, v):
        if not bm.CASE_KEY_RE.match(v):
            raise ValueError("case_key must match [a-z0-9][a-z0-9._-]{0,99}")
        return v

    @field_validator("task_type")
    @classmethod
    def _task(cls, v):
        if v not in bm.TASK_TYPES:
            raise ValueError(f"task_type must be one of {sorted(bm.TASK_TYPES)}")
        return v

    @field_validator("category")
    @classmethod
    def _cat(cls, v):
        if v not in bm.CATEGORIES:
            raise ValueError(f"category must be one of {sorted(bm.CATEGORIES)}")
        return v

    @field_validator("difficulty")
    @classmethod
    def _diff(cls, v):
        if v not in bm.DIFFICULTIES:
            raise ValueError(f"difficulty must be one of {sorted(bm.DIFFICULTIES)}")
        return v

    @field_validator("suites")
    @classmethod
    def _suites(cls, v):
        if not v or set(v) - bm.SUITES or len(set(v)) != len(v):
            raise ValueError(f"suites must be a non-empty list of unique values from {sorted(bm.SUITES)}")
        if "full" not in v:
            raise ValueError("every case must belong to the 'full' suite ('pr' is a subset of 'full')")
        return v

    _spec = field_validator("eval_spec")(_check_spec)


class BenchmarkImport(BaseModel):
    benchmark: str
    version: str
    description: str | None = None
    cases: list[BenchmarkCase] = Field(min_length=1)

    @field_validator("benchmark")
    @classmethod
    def _name(cls, v):
        if not bm.NAME_RE.match(v):
            raise ValueError("benchmark must match [a-z0-9][a-z0-9._-]{0,99}")
        return v

    @field_validator("version")
    @classmethod
    def _version(cls, v):
        if not v.strip() or len(v) > 50:
            raise ValueError("version must be a non-empty string of at most 50 characters")
        return v

    @model_validator(mode="after")
    def _unique_keys(self):
        seen, dupes = set(), set()
        for c in self.cases:
            (dupes if c.case_key in seen else seen).add(c.case_key)
        if dupes:
            raise ValueError(f"duplicate case_key(s): {sorted(dupes)}")
        return self


class BenchmarkImportResponse(BaseModel):
    benchmark: str
    version: str
    created: int
    unchanged: int
    cases: list[DatasetResponse]
