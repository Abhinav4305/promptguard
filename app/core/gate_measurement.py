"""
Controlled measurement of the release gate on SIMULATED model behaviour.

What this is: the real benchmark cases, the real executor, the real evaluators and
the real comparator, driven by hand-written model answers (benchmarks/simulations/).
It measures how the GATE behaves (aggregation, thresholds, similarity vs correctness)
on answers whose correctness is known.

What this is NOT: a measurement of any real model. No LLM is called. The answer
variants are written by hand, `wrong_close` is deliberately close in wording to the
reference, and the 'regression' / 'acceptable' labels are my judgement, verified only
against the benchmark's own correctness checks. Treat the numbers as a regression test
for the gate and a comparison of the two gating policies, not as production accuracy.

Two policies are compared on the very same runs:
  new   the current gate: gating evaluators per case (correctness first)
  old   the previous gate: text_similarity of every case, nothing else
"""

import random
from dataclasses import dataclass, field
from types import SimpleNamespace as NS
from typing import Any, Callable

import yaml
from sqlalchemy import create_engine
from sqlalchemy.orm import selectinload, sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.benchmark import case_content_hash
from app.core.comparator import ComparisonStatus, evaluate_comparison
from app.core.config import settings
from app.core.evaluators import EvalInput, create_evaluator
from app.core.run_executor import execute_run
from app.db.models import Dataset, EvaluationResult, EvaluationRun, Prompt, ResultStatus, RunStatus
from app.db.session import Base
from app.schemas.dataset import BenchmarkImport

REFUSAL = "I'm sorry, but I can't help with that."
ANSWER_KINDS = ("good", "paraphrase", "wrong_close", "wrong_far")

LABELLING_RULE = (
    "acceptable = every changed answer is a hand-written CORRECT paraphrase; "
    "regression = at least 25% of cases answered wrongly (wrong_close / wrong_far), or a whole "
    "category breaks (format_break: every structured-output answer becomes prose), or every answer "
    "is empty or a refusal. 10%-wrong scenarios are reported as a sensitivity curve only: whether "
    "they should block is a policy choice, not a fact."
)


@dataclass
class Scenario:
    name: str
    kind: str
    label: str                       # "acceptable" | "regression" | "sensitivity"
    fraction: float | None
    seed: int | None
    answer: Callable[[str], str]     # case_key -> answer
    changed_cases: int = 0


@dataclass
class _Sim:
    answers: dict[str, dict[str, str]]


class _ScriptedLLM:
    def __init__(self, by_query: dict[str, str]):
        self.by_query = by_query

    def __call__(self, model_name, system_prompt, user_query, temperature=0.0):
        return NS(output=self.by_query[user_query], latency_seconds=0.5, token_cost=0.0,
                  prompt_tokens=1, completion_tokens=1)


def build_scenarios(case_keys: list[str], structured_keys: set[str], sim: dict[str, dict[str, str]],
                    suite: str, seeds: int) -> list[Scenario]:
    keys = sorted(case_keys)
    n = len(keys)
    out: list[Scenario] = []

    def subset(kind: str, fraction: float, seed: int) -> set[str]:
        k = n if fraction >= 1.0 else max(1, round(fraction * n))
        return set(random.Random(f"{suite}|{kind}|{fraction}|{seed}").sample(keys, k))

    def mixed(kind: str, answer_kind: str, fraction: float, label: str):
        for seed in range(1 if fraction >= 1.0 else seeds):
            chosen = subset(kind, fraction, seed)
            out.append(Scenario(
                name=f"{kind}@{int(fraction * 100)}%#{seed}", kind=kind, label=label, fraction=fraction, seed=seed,
                answer=lambda key, chosen=chosen, ak=answer_kind: sim[key][ak] if key in chosen else sim[key]["good"],
                changed_cases=len(chosen),
            ))

    for f in (0.25, 0.5, 1.0):
        mixed("paraphrase", "paraphrase", f, "acceptable")
    for kind in ("wrong_close", "wrong_far"):
        mixed(kind, kind, 0.1, "sensitivity")
        for f in (0.25, 0.5, 1.0):
            mixed(kind, kind, f, "regression")

    # Exactly one wrong case: shows how small a change the aggregate thresholds can see.
    for seed in range(seeds):
        one = {random.Random(f"{suite}|single|{seed}").choice(keys)}
        out.append(Scenario(
            name=f"single_wrong_close#{seed}", kind="single_wrong_close", label="sensitivity", fraction=None, seed=seed,
            answer=lambda key, one=one: sim[key]["wrong_close"] if key in one else sim[key]["good"],
            changed_cases=1,
        ))

    structured = set(structured_keys) & set(keys)
    if structured:
        out.append(Scenario("format_break", "format_break", "regression", None, None,
                            lambda key: sim[key]["wrong_far"] if key in structured else sim[key]["good"],
                            changed_cases=len(structured)))
    out.append(Scenario("empty_output", "empty_output", "regression", None, None, lambda key: "", n))
    out.append(Scenario("refusal", "refusal", "regression", None, None, lambda key: REFUSAL, n))
    return out


# ── the two gating policies ──────────────────────────────────────────────────────────────

def _legacy_view(results, expected_by_dataset: dict[int, str]):
    """What the previous pipeline stored: one text_similarity score per case, gating, nothing else."""
    ts = create_evaluator("text_similarity")
    view = []
    for r in results:
        score = ts.evaluate(EvalInput(r.actual_output or "", expected_by_dataset[r.dataset_id], {})).score
        view.append(NS(
            dataset_id=r.dataset_id, status=r.status, error_category=r.error_category,
            latency_seconds=r.latency_seconds, token_cost=r.token_cost, actual_output=r.actual_output,
            scores=[NS(evaluator_name="text_similarity", evaluator_version=ts.version, kind="similarity",
                       config_hash=ts.config_hash, gating=True, status="OK", score=score)],
        ))
    return view


def _rate(k: int, n: int) -> dict[str, Any]:
    return {"count": k, "of": n, "rate": round(k / n, 4) if n else None}


def measure(benchmark_path: str, simulation_path: str, *, seeds: int = 10,
            suites: tuple[str, ...] = ("pr", "full")) -> dict[str, Any]:
    with open(benchmark_path, encoding="utf-8") as f:
        bench = BenchmarkImport.model_validate(yaml.safe_load(f))
    with open(simulation_path, encoding="utf-8") as f:
        sim = yaml.safe_load(f)
    missing = [c.case_key for c in bench.cases if c.case_key not in sim or set(ANSWER_KINDS) - set(sim[c.case_key])]
    if missing:
        raise ValueError(f"simulation lacks answers for cases: {missing}")

    engine = create_engine("sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine, autocommit=False, autoflush=False)()
    try:
        prompt = Prompt(version_tag="sim", prompt_text="simulated", model_name="simulated")
        db.add(prompt)
        rows: dict[str, Dataset] = {}
        for c in bench.cases:
            h = case_content_hash(input_query=c.input_query, expected_output=c.expected_output,
                                  eval_spec=c.eval_spec, task_type=c.task_type, category=c.category,
                                  difficulty=c.difficulty, suites=c.suites)
            rows[c.case_key] = Dataset(
                input_query=c.input_query, expected_output=c.expected_output, benchmark=bench.benchmark,
                benchmark_version=bench.version, case_key=c.case_key, task_type=c.task_type,
                category=c.category, difficulty=c.difficulty, suites=c.suites, eval_spec=c.eval_spec,
                content_hash=h)
        db.add_all(rows.values())
        db.commit()
        expected_by_id = {d.id: d.expected_output for d in rows.values()}
        meta = {c.case_key: c for c in bench.cases}

        def run_suite(keys: list[str], answer: Callable[[str], str]):
            run = EvaluationRun(prompt_id=prompt.id, status=RunStatus.PENDING, expected_cases=len(keys))
            db.add(run)
            db.flush()
            db.add_all(EvaluationResult(run_id=run.id, dataset_id=rows[k].id, status=ResultStatus.PENDING) for k in keys)
            db.commit()
            llm = _ScriptedLLM({rows[k].input_query: answer(k) for k in keys})
            execute_run(db, run.id, llm_fn=llm)
            db.expire_all()
            results = (db.query(EvaluationResult).options(selectinload(EvaluationResult.scores))
                       .filter_by(run_id=run.id).order_by(EvaluationResult.dataset_id).all())
            return db.get(EvaluationRun, run.id), results

        report: dict[str, Any] = {
            "simulated": True,
            "disclaimer": (
                "SIMULATED. Hand-written model answers replayed through the real executor, evaluators and "
                "comparator. No LLM was called, so these are NOT measurements of llama3.2 or any real model. "
                "wrong_close answers are deliberately close to the reference wording, which favours the "
                "correctness gate over the similarity gate; read the wrong_far rows for the opposite case."
            ),
            "labelling_rule": LABELLING_RULE,
            "benchmark": {"name": bench.benchmark, "version": bench.version, "cases": len(bench.cases)},
            "thresholds": {
                "similarity_drop": settings.regression_similarity_drop_threshold,
                "correctness_drop": settings.regression_correctness_drop_threshold,
            },
            "seeds_per_mixed_scenario": seeds,
            "suites": {},
        }

        for suite in suites:
            keys = sorted(c.case_key for c in bench.cases if suite in c.suites)
            structured = {k for k in keys if meta[k].task_type == "structured_output"}
            base_run, base_results = run_suite(keys, lambda k: sim[k]["good"])
            base_legacy = _legacy_view(base_results, expected_by_id)
            scenarios = build_scenarios(keys, structured, sim, suite, seeds)

            rows_out = []
            for sc in scenarios:
                cand_run, cand_results = run_suite(keys, sc.answer)
                new = evaluate_comparison(base_run, base_results, cand_run, cand_results)
                old = evaluate_comparison(base_run, base_legacy, cand_run, _legacy_view(cand_results, expected_by_id))
                rows_out.append({
                    "scenario": sc.name, "kind": sc.kind, "label": sc.label, "fraction": sc.fraction,
                    "seed": sc.seed, "cases_changed": sc.changed_cases, "cases_total": len(keys),
                    "new_blocked": not new.passed, "new_status": new.status.value,
                    "new_flagged": [e.name for e in new.report.evaluators if e.regression_detected] if new.report else [],
                    "old_blocked": not old.passed, "old_status": old.status.value,
                    "cases_scored_lower": len({c.dataset_id for c in new.report.regressed_cases}) if new.report else None,
                })
            bad = [r for r in rows_out if r["new_status"] != ComparisonStatus.OK.value or r["old_status"] != ComparisonStatus.OK.value]
            if bad:
                raise RuntimeError(f"unexpected non-OK comparison in simulation: {bad[:3]}")

            def tally(label: str, gate: str, kinds: set[str] | None = None):
                sel = [r for r in rows_out if r["label"] == label and (kinds is None or r["kind"] in kinds)]
                return _rate(sum(r[f"{gate}_blocked"] for r in sel), len(sel))

            by_scenario_type: dict[str, Any] = {}
            for r in rows_out:
                key = f'{r["kind"]}@{int(r["fraction"] * 100)}%' if r["fraction"] is not None else r["kind"]
                slot = by_scenario_type.setdefault(key, {"label": r["label"], "new_blocked": 0, "old_blocked": 0, "of": 0})
                slot["new_blocked"] += r["new_blocked"]
                slot["old_blocked"] += r["old_blocked"]
                slot["of"] += 1

            report["suites"][suite] = {
                "cases": len(keys),
                "scenarios_run": len(rows_out),
                "headline": {
                    "detection_of_regressions": {"new": tally("regression", "new"), "old_similarity_only": tally("regression", "old")},
                    "false_blocks_on_acceptable_changes": {"new": tally("acceptable", "new"), "old_similarity_only": tally("acceptable", "old")},
                },
                "by_scenario_type": by_scenario_type,
                "scenarios": rows_out,
            }
        return report
    finally:
        db.close()
        engine.dispose()


def format_markdown(report: dict[str, Any]) -> str:
    lines = [f"> {report['disclaimer']}", "", f"Labelling: {report['labelling_rule']}", ""]
    for suite, s in report["suites"].items():
        h = s["headline"]
        lines += [f"### Suite `{suite}` ({s['cases']} cases, {s['scenarios_run']} simulated candidates)", "",
                  "| Gate | Regressions blocked | Acceptable changes blocked (false blocks) |", "|---|---|---|"]
        for gate, label in (("new", "new (correctness-first)"), ("old_similarity_only", "old (similarity only)")):
            d, f = h["detection_of_regressions"][gate], h["false_blocks_on_acceptable_changes"][gate]
            lines.append(f"| {label} | {d['count']}/{d['of']} ({d['rate']:.0%}) | {f['count']}/{f['of']} ({f['rate']:.0%}) |")
        lines += ["", "| Scenario type | Label | New gate blocked | Old gate blocked |", "|---|---|---|---|"]
        for name, v in s["by_scenario_type"].items():
            lines.append(f"| {name} | {v['label']} | {v['new_blocked']}/{v['of']} | {v['old_blocked']}/{v['of']} |")
        lines.append("")
    return "\n".join(lines)
