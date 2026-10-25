from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.benchmark import case_content_hash
from app.db.models.dataset import Dataset
from app.db.session import get_db
from app.schemas.dataset import (
    BenchmarkImport, BenchmarkImportResponse, DatasetCreate, DatasetResponse,
)

router = APIRouter(prefix="/datasets", tags=["datasets"])

@router.post("/", response_model=DatasetResponse, status_code=201)
def create_dataset(payload: DatasetCreate, db: Session = Depends(get_db)):
    dataset = Dataset(**payload.model_dump())
    db.add(dataset)
    db.commit()
    db.refresh(dataset)
    return dataset

@router.post("/bulk", response_model=list[DatasetResponse], status_code=201)
def create_datasets_bulk(payload: list[DatasetCreate], db: Session = Depends(get_db)):
    datasets = [Dataset(**item.model_dump()) for item in payload]
    db.add_all(datasets)
    db.commit()
    for d in datasets:
        db.refresh(d)
    return datasets


@router.post("/import", response_model=BenchmarkImportResponse)
def import_benchmark(payload: BenchmarkImport, db: Session = Depends(get_db)):
    """
    Idempotently load a versioned benchmark. Case identity is
    (benchmark, version, case_key).

      * new case                      -> created
      * same case, identical content  -> returned unchanged (safe to re-run in CI)
      * same case, different content  -> HTTP 409 and NOTHING is written: a published
        benchmark version is immutable. Publish the change as a new version.

    The response lists the cases in payload order; filter them by `suites` to pick
    the dataset ids to evaluate.
    """
    existing = {
        d.case_key: d
        for d in db.query(Dataset).filter(
            Dataset.benchmark == payload.benchmark, Dataset.benchmark_version == payload.version
        )
    }
    hashes = {
        c.case_key: case_content_hash(
            input_query=c.input_query, expected_output=c.expected_output, eval_spec=c.eval_spec,
            task_type=c.task_type, category=c.category, difficulty=c.difficulty, suites=c.suites,
        )
        for c in payload.cases
    }
    changed = sorted(k for k, d in existing.items() if k in hashes and d.content_hash != hashes[k])
    if changed:
        raise HTTPException(
            status_code=409,
            detail=(
                f"benchmark {payload.benchmark!r} version {payload.version!r} already has different "
                f"content for case(s) {changed}. A published version is immutable: publish the "
                f"change under a new version."
            ),
        )

    rows, created = [], 0
    for c in payload.cases:
        row = existing.get(c.case_key)
        if row is None:
            row = Dataset(
                input_query=c.input_query, expected_output=c.expected_output,
                benchmark=payload.benchmark, benchmark_version=payload.version, case_key=c.case_key,
                task_type=c.task_type, category=c.category, difficulty=c.difficulty,
                suites=c.suites, eval_spec=c.eval_spec, content_hash=hashes[c.case_key],
            )
            db.add(row)
            created += 1
        rows.append(row)
    try:
        db.commit()
    except IntegrityError:   # a concurrent import of the same version won the race
        db.rollback()
        raise HTTPException(status_code=409, detail="benchmark is being imported concurrently; retry") from None
    for r in rows:
        db.refresh(r)
    return BenchmarkImportResponse(
        benchmark=payload.benchmark, version=payload.version,
        created=created, unchanged=len(rows) - created, cases=rows,
    )


@router.get("/", response_model=list[DatasetResponse])
def list_datasets(
    benchmark: str | None = Query(None),
    benchmark_version: str | None = Query(None),
    suite: str | None = Query(None, description="e.g. 'pr' or 'full'"),
    category: str | None = Query(None),
    db: Session = Depends(get_db),
):
    q = db.query(Dataset)
    if benchmark is not None:
        q = q.filter(Dataset.benchmark == benchmark)
    if benchmark_version is not None:
        q = q.filter(Dataset.benchmark_version == benchmark_version)
    if category is not None:
        q = q.filter(Dataset.category == category)
    rows = q.order_by(Dataset.id).all()
    if suite is not None:   # JSON array membership, done in Python to stay database-portable
        rows = [r for r in rows if r.suites and suite in r.suites]
    return rows

@router.get("/{dataset_id}", response_model=DatasetResponse)
def get_dataset(dataset_id: int, db: Session = Depends(get_db)):
    dataset = db.get(Dataset, dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return dataset
