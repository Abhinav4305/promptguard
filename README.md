# PromptGuard

**Lightweight LLM Regression Detection Platform**

PromptGuard automatically catches prompt and model changes that silently degrade AI performance before they reach production. It plugs into a standard CI/CD pipeline and fails the build if a new prompt version is worse than the baseline across accuracy, latency, or cost.

---

## The Problem

When you tweak a prompt, swap an LLM model, or adjust generation settings, there is no automated way to know whether the change made the AI better or quietly worse. Unlike regular code bugs that crash visibly, a bad prompt change just produces slightly degraded answers and nothing in a standard CI pipeline catches that.

## What PromptGuard Does

- Stores **versioned prompts** and **test datasets** in a database
- Runs **asynchronous evaluations** through a Celery task queue, calling the LLM for every test question and recording the actual response
- Scores each response against expected output using **ROUGE-L** and **Levenshtein similarity** — no LLM-as-judge, no API costs for evaluation itself
- Captures **latency** and **token cost** per call
- Compares a new run against a **baseline run** across all three axes and applies configurable thresholds to produce a pass/fail verdict
- **Fails the CI build** automatically on regression via a GitHub Actions workflow
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
│   │       ├── datasets.py         # CRUD for test cases
│   │       ├── evaluations.py      # Trigger runs, poll status, compare
│   │       └── dashboard.py        # Visual HTML comparison dashboard
│   ├── core/
│   │   ├── config.py               # Settings from environment variables
│   │   ├── llm_gateway.py          # Ollama REST API wrapper
│   │   ├── metrics.py              # ROUGE-L + Levenshtein scoring
│   │   └── comparator.py           # Baseline vs candidate delta engine
│   ├── db/
│   │   ├── session.py              # SQLAlchemy engine + session
│   │   └── models/                 # Prompt, Dataset, EvaluationRun, EvaluationResult
│   ├── schemas/                    # Pydantic request/response models
│   ├── tasks/
│   │   ├── celery_app.py           # Celery instance
│   │   └── evaluation_tasks.py     # Async LLM evaluation loop
│   └── main.py                     # FastAPI app entrypoint
├── .github/
│   └── workflows/
│       └── promptguard-ci.yml      # GitHub Actions regression check
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── .env.example
```

---

## Database Schema

| Table | Purpose |
|---|---|
| `prompts` | Versioned system instructions with model name |
| `datasets` | Evaluation test cases (input query + expected output) |
| `evaluation_runs` | Master record per test run with status and baseline flag |
| `evaluation_results` | Per-question metrics: similarity score, latency, token cost, actual output |

---

## Regression Thresholds

| Metric | Default Threshold | Direction |
|---|---|---|
| Similarity | > 10% drop | Lower is worse |
| Latency | > 50% spike | Higher is worse |
| Cost | > 20% spike | Higher is worse |

All thresholds are configurable via environment variables.

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
# 1. Create a prompt
curl -X POST http://localhost:8000/prompts/ \
  -H "Content-Type: application/json" \
  -d '{"version_tag": "v1", "prompt_text": "You are a concise assistant. Answer in one sentence.", "model_name": "llama3.2"}'

# 2. Create test cases
curl -X POST http://localhost:8000/datasets/bulk \
  -H "Content-Type: application/json" \
  -d '[
    {"input_query": "What is 2 + 2?", "expected_output": "2 + 2 equals 4."},
    {"input_query": "What is the capital of France?", "expected_output": "The capital of France is Paris."}
  ]'

# 3. Trigger a baseline evaluation
curl -X POST http://localhost:8000/evaluations/run \
  -H "Content-Type: application/json" \
  -d '{"prompt_id": 1, "dataset_ids": [1, 2], "is_baseline": true}'

# 4. Poll until COMPLETED
curl http://localhost:8000/evaluations/1

# 5. Mark as official baseline
curl -X PATCH http://localhost:8000/evaluations/1/set-baseline

# 6. Run a candidate evaluation
curl -X POST http://localhost:8000/evaluations/run \
  -H "Content-Type: application/json" \
  -d '{"prompt_id": 2, "dataset_ids": [1, 2]}'

# 7. Compare via JSON API
curl "http://localhost:8000/evaluations/compare?baseline_run_id=1&candidate_run_id=2"

# 8. Open the visual dashboard in your browser
open http://localhost:8000/dashboard/compare?baseline_run_id=1&candidate_run_id=2
```

---

## Visual Dashboard

Open `/dashboard/compare?baseline_run_id=X&candidate_run_id=Y` in any browser to see:

- Pass/fail verdict banner
- Metric cards showing similarity, latency, and cost with baseline to candidate deltas
- Bar chart comparing similarity scores per question side by side
- Per-question breakdown showing the exact baseline and candidate answers

---

## CI/CD Pipeline

The GitHub Actions workflow runs automatically on every push to `main` and on every pull request. It:

1. Installs Ollama and pulls `llama3.2` on a fresh Ubuntu runner
2. Starts the full Docker stack
3. Seeds a baseline prompt and evaluation datasets
4. Runs a baseline evaluation and marks it
5. Runs a candidate evaluation
6. Calls `/evaluations/compare` and fails the build if `passed` is `false`
7. Posts the full comparison JSON to the GitHub Actions job summary

A regression in any metric blocks the merge.

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/prompts/` | Create a versioned prompt |
| `GET` | `/prompts/` | List all prompts |
| `POST` | `/datasets/bulk` | Create multiple test cases at once |
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
| `REGRESSION_SIMILARITY_DROP_THRESHOLD` | Max allowed similarity drop | `0.10` |
| `REGRESSION_LATENCY_SPIKE_THRESHOLD` | Max allowed latency increase | `0.50` |
| `REGRESSION_COST_SPIKE_THRESHOLD` | Max allowed cost increase | `0.20` |

---

## Built With

FastAPI · PostgreSQL · SQLAlchemy · Celery · Redis · Ollama · Docker · GitHub Actions

---

*Built as a portfolio project demonstrating production-grade MLOps tooling — automated LLM quality regression detection integrated into a CI/CD pipeline.*
