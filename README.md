# PromptGuard

**Lightweight LLM Regression Detection Platform**

PromptGuard automatically catches prompt and model changes that silently degrade AI performance before they reach production. It plugs into a standard CI/CD pipeline and fails the build if a new prompt version is worse than the baseline across accuracy, latency, or cost.

---

## The Problem

When you tweak a prompt, swap an LLM model, or adjust generation settings, there is no automated way to know whether the change made the AI better or quietly worse. Unlike regular code bugs that crash visibly, a bad prompt change just produces slightly degraded answers and nothing in a standard CI pipeline catches that.

## What PromptGuard Does

- Stores **versioned prompts** and **test datasets** in a database
- Runs **asynchronous evaluations** through a Celery task queue, calling the LLM for every test question and recording the actual response
- Scores each response with **pluggable, versioned evaluators**: objective checks (exact match, required/forbidden terms, JSON schema validity, JSON equality) plus the original **ROUGE-L / Levenshtein similarity**. Correctness checks gate a release; similarity is an indicator next to them. No LLM-as-judge yet (see [docs/evaluators.md](docs/evaluators.md))
- Captures **latency** and **token cost** per call
- Compares a new run against a **baseline run** per evaluator, latency and cost, and applies configurable thresholds to produce a pass/fail verdict. Runs with failed cases or different evaluators are reported as `INCOMPLETE` / `INCOMPARABLE`, never as a quality verdict
- Ships a **versioned benchmark** (`benchmarks/core-v1.yaml`: 39 cases, 12 in a fast `pr` suite) with stable case IDs, categories and immutable definitions ([docs/benchmarks.md](docs/benchmarks.md))
- **Fails the CI build** automatically on regression via a GitHub Actions workflow. Making that failure block merges requires a branch protection rule on `main` that requires the `regression-check` job; that is a repository setting and is not verified or configured by this code
- Serves a **visual comparison dashboard** at `/dashboard/compare` showing bar charts and per-question answer breakdowns

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend API | FastAPI (Python) |
| Database | PostgreSQL + SQLAlchemy |
| Task Queue | Celery + Redis |
| LLM (local) | Ollama (llama3.2) |
| Evaluation Metrics | ROUGE-L, Levenshtein |
| Containerisation | Docker + Docker Compose |
| CI/CD | GitHub Actions |

---

## Project Structure

```
promptguard/
├── app/
│   ├── api/
│   │   └── routes/
│   │       ├── prompts.py          # CRUD for versioned prompts
│   │       ├── datasets.py         # Test cases, versioned benchmark import (POST /datasets/import)
│   │       ├── evaluations.py      # Trigger runs, poll status, compare
│   │       └── dashboard.py        # Visual HTML comparison dashboard
│   ├── core/
│   │   ├── config.py               # Settings from environment variables
│   │   ├── llm_gateway.py          # Ollama REST API wrapper
│   │   ├── metrics.py              # ROUGE-L + Levenshtein scoring (used by text_similarity)
│   │   ├── evaluators/             # Evaluator interface, registry, built-in evaluators, gating rule
│   │   ├── benchmark.py            # Benchmark validation and case content hash
│   │   ├── run_executor.py         # Failure-aware execution of one run
│   │   ├── comparator.py           # Comparability checks + baseline vs candidate delta engine
│   │   └── gate_measurement.py     # Simulated gate measurement (no LLM)
│   ├── db/
│   │   ├── session.py              # SQLAlchemy engine + session
│   │   └── models/                 # Prompt, Dataset, EvaluationRun, EvaluationResult, EvaluationScore
│   ├── schemas/                    # Pydantic request/response models
│   ├── tasks/
│   │   ├── celery_app.py           # Celery instance
│   │   └── evaluation_tasks.py     # Async LLM evaluation loop
│   └── main.py                     # FastAPI app entrypoint
├── alembic/                        # Database migrations
├── benchmarks/
│   ├── core-v1.yaml                # Versioned benchmark (pr + full suites)
│   └── simulations/                # Hand-written answers used by the gate measurement
├── docs/                           # evaluators.md, benchmarks.md, gate-measurement.md
├── scripts/
│   ├── load_benchmark.py           # Load a benchmark into a running API, print suite dataset ids
│   └── measure_gate.py             # Controlled, simulated gate measurement
├── tests/
├── prompts/
│   └── production.yaml             # Source of truth for the currently "live" prompt
├── .github/
│   └── workflows/
│       └── promptguard-ci.yml      # GitHub Actions regression check
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── seed.py
└── .env.example
```

---

## Database Schema

| Table | Purpose |
|---|---|
| `prompts` | Versioned system instructions with model name |
| `datasets` | Evaluation test cases (input query + expected output); benchmark cases also carry `benchmark`, `benchmark_version`, `case_key`, `task_type`, `category`, `difficulty`, `suites`, `eval_spec`, `content_hash` |
| `evaluation_runs` | Master record per test run with status and baseline flag |
| `evaluation_results` | Per-question execution record: status, latency, token cost, actual output (`similarity_score` is a compatibility copy of the `text_similarity` score) |
| `evaluation_scores` | One row per (result, evaluator): evaluator name, version, kind, config and hash, `gating` flag, score or `ERROR`, raw details |

---

## Regression Thresholds

| Metric | Default Threshold | Direction |
|---|---|---|
| Similarity evaluators (gating) | > 10% drop of the mean | Lower is worse |
| Correctness evaluators (gating) | > 10% drop of the mean | Lower is worse |
| Latency | > 50% spike | Higher is worse |
| Cost | > 20% spike | Higher is worse |

Drops are relative to the baseline mean, per evaluator. Non-gating evaluators are reported but cannot fail the verdict. All thresholds are configurable via environment variables.

---

## Run Statuses and Comparison Policy

An evaluation run is **COMPLETED only if every test case executed successfully**. COMPLETED describes execution, never quality; quality verdicts only exist in a comparison.

| Run status | Meaning |
|---|---|
| `PENDING` / `RUNNING` | Queued / executing (a run stays `RUNNING` between retry attempts) |
| `COMPLETED` | All expected cases succeeded (`failed_cases == 0`) |
| `PARTIAL` | Some cases failed. Results are kept for debugging; the run cannot be a baseline |
| `FAILED` | No case succeeded, or the run could not start |

Each case has its own `status` (`PENDING`, `RUNNING`, `SUCCEEDED`, `FAILED`). A failed case records an `error_category` and `error_message`: `INFRASTRUCTURE` (timeout, connection, HTTP 5xx/429; retried), `MODEL_ERROR` (request rejected or malformed reply), `EVALUATOR_ERROR` (scoring failed; the model output is kept) or `UNKNOWN`. Quality regressions are never an error category.

`GET /evaluations/compare` first checks the two runs are comparable and returns a `comparison_status`:

| `comparison_status` | Meaning | `passed` |
|---|---|---|
| `OK` | Same cases, all executed, every score present, same evaluators. `passed` is the threshold verdict | per thresholds |
| `INCOMPLETE` | A run has failed, unfinished cases, or a missing / errored / invalid evaluator score | always `false` |
| `INCOMPARABLE` | The runs do not cover the same case IDs, contain duplicates, or were scored by different evaluators (name, version, config or gating) | always `false` |

**Policy (`comparison-policy-v2`): incomplete or incomparable evaluations block the release.** No aggregate score is reported for them, because an average over a subset of cases is not a measurement of the benchmark. `issues` lists exactly which cases and why.

## Evaluators and the benchmark

* [docs/evaluators.md](docs/evaluators.md): the evaluator contract, built-in evaluators, which scores can block a release, how to add one, and an outline for calibrating a future LLM judge.
* [docs/benchmarks.md](docs/benchmarks.md): file format, suites, loading, immutability, and the honest limits of `core-v1`.
* [docs/gate-measurement.md](docs/gate-measurement.md): detection and false-block rates of the gate on **simulated** answers (`python scripts/measure_gate.py`). Not a measurement of any real model.

## Migrations and Tests

The schema is managed by Alembic (`alembic/versions/`); the API applies pending migrations at startup. A database created by an older version (no `alembic_version` table) is stamped at the baseline revision and then upgraded, which also corrects old runs that were recorded as COMPLETED despite failed cases (they become `PARTIAL`/`FAILED`).

```bash
pip install -r requirements-dev.txt
pytest                                   # unit + API + migration tests on SQLite, no services needed
TEST_POSTGRES_URL=postgresql://user:pass@localhost:5432/scratch pytest tests/test_migrations.py   # also runs the Postgres tests (WIPES that database's public schema)
```

---

## Getting Started

### Prerequisites

- Docker Desktop
- Ollama installed and running (`ollama serve`)
- `llama3.2` model pulled (`ollama pull llama3.2`)

### Setup

```bash
git clone https://github.com/Abhinav4305/promptguard.git
cd promptguard

cp .env.example .env

docker compose up --build
```

The API will be available at `http://localhost:8000`. Interactive API docs at `http://localhost:8000/docs`.

### Running an Evaluation

```bash
# Start the stack
docker compose up --build -d

# Run the seed script (installs httpx + pyyaml if needed, then seeds everything)
pip install httpx pyyaml
python seed.py
```

The seed script reads the baseline prompt straight from `prompts/production.yaml`, so the local demo always reflects whatever prompt is actually considered "live." It then seeds a deliberately worse candidate prompt so the dashboard has a real regression to show, runs both evaluations, and prints a dashboard URL when it finishes. Open it in your browser to see the comparison.

---

## Changing the Live Prompt

`prompts/production.yaml` is the single source of truth for the prompt currently in use. To propose a change:

1. Edit `prompts/production.yaml` on a new branch
2. Open a pull request against `main`
3. CI automatically evaluates your candidate prompt against whatever `main` currently has, and reports a pass/fail regression check on the PR
4. If it passes, merge — that version becomes the new baseline for the next person's change

---

## Visual Dashboard

Open `/dashboard/compare?baseline_run_id=X&candidate_run_id=Y` in any browser to see:

- Pass/fail verdict banner
- Metric cards per evaluator (advisory ones marked), plus latency and cost, with baseline to candidate deltas
- A note when individual cases scored lower but the aggregate stayed within threshold
- Bar chart of the gating score per question (mean of the gating evaluators), baseline vs candidate
- Per-question breakdown with each evaluator's score and the exact baseline and candidate answers
- For `INCOMPLETE` / `INCOMPARABLE` comparisons: the reasons and failed cases instead of a verdict

---

## CI/CD Pipeline

The GitHub Actions workflow runs automatically on every push to `main` and on every pull request, and behaves differently depending on which:

**On a pull request** — comparing a proposed change against what's currently live:
1. Fetches `main` and reads `prompts/production.yaml` from both the PR branch (candidate) and `main` (baseline)
2. Installs Ollama, pulls `llama3.2`, and starts the full Docker stack
3. Seeds both prompt versions and loads the PR suite of the versioned benchmark (`scripts/load_benchmark.py`; fails if a published benchmark version was edited in place)
4. Runs a baseline evaluation and a candidate evaluation
5. Calls `/evaluations/compare` and fails the build if `passed` is `false`
6. Posts the full comparison JSON to the GitHub Actions job summary

**On a direct push to `main`** (typically a freshly merged PR) — establishing the new baseline:
1. Reads `prompts/production.yaml` at the new commit
2. Runs a single evaluation against it and marks the result the new official baseline — no comparison step, since there's nothing to compare a fresh merge against

The job fails (red check) on a detected regression and on `INCOMPLETE` / `INCOMPARABLE` comparisons. Whether that blocks merging depends on a branch protection rule on `main` requiring the `regression-check` job; configure it in the repository settings, this repo cannot verify it. Not implemented yet: a stored JSON report and uploaded artifacts.

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/prompts/` | Create a versioned prompt |
| `GET` | `/prompts/` | List all prompts |
| `POST` | `/datasets/bulk` | Create multiple test cases at once (optional `eval_spec` per case) |
| `POST` | `/datasets/import` | Idempotently import a versioned benchmark (409 if a published case changed) |
| `GET` | `/datasets/` | List cases; filter by `benchmark`, `benchmark_version`, `suite`, `category` |
| `POST` | `/evaluations/run` | Trigger an async evaluation run |
| `GET` | `/evaluations/{id}` | Poll run status and results |
| `PATCH` | `/evaluations/{id}/set-baseline` | Mark a completed run as the baseline |
| `GET` | `/evaluations/compare` | JSON regression comparison report |
| `GET` | `/dashboard/compare` | Visual HTML regression dashboard |
| `GET` | `/health` | Health check |
| `GET` | `/docs` | Interactive Swagger UI |

---

## Environment Variables

| Variable | Description | Default |
|---|---|---|
| `DATABASE_URL` | PostgreSQL connection string | `postgresql://promptguard:promptguard@db:5432/promptguard` |
| `REDIS_URL` | Redis connection string | `redis://redis:6379/0` |
| `CELERY_BROKER_URL` | Celery broker URL | `redis://redis:6379/0` |
| `CELERY_RESULT_BACKEND` | Celery result backend | `redis://redis:6379/0` |
| `REGRESSION_SIMILARITY_DROP_THRESHOLD` | Max allowed drop of a gating similarity evaluator's mean | `0.10` |
| `REGRESSION_CORRECTNESS_DROP_THRESHOLD` | Max allowed drop of a gating correctness evaluator's mean | `0.10` |
| `REGRESSION_LATENCY_SPIKE_THRESHOLD` | Max allowed latency increase | `0.50` |
| `REGRESSION_COST_SPIKE_THRESHOLD` | Max allowed cost increase | `0.20` |

---

## Built With

FastAPI · PostgreSQL · SQLAlchemy · Celery · Redis · Ollama · Docker · GitHub Actions

---

*Built as a portfolio project demonstrating production-grade MLOps tooling — automated LLM quality regression detection integrated into a CI/CD pipeline.*