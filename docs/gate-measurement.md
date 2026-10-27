# Gate measurement on controlled, simulated answers

**Read this first.** Nothing below measures a real model. No LLM was called. The numbers describe how the
*release gate* (aggregation, thresholds, similarity vs correctness checks) behaves on hand-written model
answers whose correctness is known. Do not quote them as "PromptGuard catches X% of regressions".

## What was run

```bash
python scripts/measure_gate.py --out gate-report.json     # seeds=10, suites pr + full, ~40 s, no services needed
```

* Cases: `benchmarks/core-v2.yaml` (65 cases; the `pr` suite is a 16-case subset).
* Answers: `benchmarks/simulations/core-v2.sim.yaml`, four hand-written answers per case
  (`good`, `paraphrase`, `wrong_close`, `wrong_far`). `tests/test_gate_measurement.py` checks that every
  label agrees with the benchmark's own correctness checks.
* Pipeline: the real executor, the real evaluators and the real comparator, on an in-memory SQLite database.
* Two gates are applied to the *same* candidate runs:
  **new** = the current gate (gating evaluators per case; correctness first),
  **old** = the previous gate (`text_similarity` of every case, nothing else).
* Thresholds: similarity drop 10%, correctness drop 10% (relative, on the mean; a drop must be *larger* than the threshold to block).
* Mixed scenarios pick which cases change with a seeded RNG (seeds 0-9), so the run is reproducible.
* Generated on 2026-10-25 on branch `quality-metric-and-benchmark-v2`. Re-run it after changing the benchmark,
  the simulation file or a threshold; the numbers in this file go stale otherwise.

## Results

> SIMULATED. Hand-written model answers replayed through the real executor, evaluators and comparator. No LLM was called, so these are NOT measurements of llama3.2 or any real model. wrong_close answers are deliberately close to the reference wording, which favours the correctness gate over the similarity gate; read the wrong_far rows for the opposite case.

Labelling: acceptable = every changed answer is a hand-written CORRECT paraphrase; regression = at least 25% of cases answered wrongly (wrong_close / wrong_far), or a whole category breaks (format_break: every structured-output answer becomes prose), or every answer is empty or a refusal. 10%-wrong scenarios are reported as a sensitivity curve only: whether they should block is a policy choice, not a fact.

### Suite `pr` (16 cases, 96 simulated candidates)

| Gate | Regressions blocked | Acceptable changes blocked (false blocks) |
|---|---|---|
| new (correctness-first) | 45/45 (100%) | 0/21 (0%) |
| old (similarity only) | 37/45 (82%) | 19/21 (90%) |

| Scenario type | Label | New gate blocked | Old gate blocked |
|---|---|---|---|
| paraphrase@25% | acceptable | 0/10 | 8/10 |
| paraphrase@50% | acceptable | 0/10 | 10/10 |
| paraphrase@100% | acceptable | 0/1 | 1/1 |
| wrong_close@10% | sensitivity | 10/10 | 0/10 |
| wrong_close@25% | regression | 10/10 | 4/10 |
| wrong_close@50% | regression | 10/10 | 9/10 |
| wrong_close@100% | regression | 1/1 | 1/1 |
| wrong_far@10% | sensitivity | 10/10 | 4/10 |
| wrong_far@25% | regression | 10/10 | 10/10 |
| wrong_far@50% | regression | 10/10 | 10/10 |
| wrong_far@100% | regression | 1/1 | 1/1 |
| single_wrong_close | sensitivity | 2/10 | 0/10 |
| format_break | regression | 1/1 | 0/1 |
| empty_output | regression | 1/1 | 1/1 |
| refusal | regression | 1/1 | 1/1 |

### Suite `full` (65 cases, 96 simulated candidates)

| Gate | Regressions blocked | Acceptable changes blocked (false blocks) |
|---|---|---|
| new (correctness-first) | 45/45 (100%) | 0/21 (0%) |
| old (similarity only) | 41/45 (91%) | 18/21 (86%) |

| Scenario type | Label | New gate blocked | Old gate blocked |
|---|---|---|---|
| paraphrase@25% | acceptable | 0/10 | 7/10 |
| paraphrase@50% | acceptable | 0/10 | 10/10 |
| paraphrase@100% | acceptable | 0/1 | 1/1 |
| wrong_close@10% | sensitivity | 10/10 | 0/10 |
| wrong_close@25% | regression | 10/10 | 7/10 |
| wrong_close@50% | regression | 10/10 | 10/10 |
| wrong_close@100% | regression | 1/1 | 1/1 |
| wrong_far@10% | sensitivity | 10/10 | 0/10 |
| wrong_far@25% | regression | 10/10 | 10/10 |
| wrong_far@50% | regression | 10/10 | 10/10 |
| wrong_far@100% | regression | 1/1 | 1/1 |
| single_wrong_close | sensitivity | 0/10 | 0/10 |
| format_break | regression | 1/1 | 0/1 |
| empty_output | regression | 1/1 | 1/1 |
| refusal | regression | 1/1 | 1/1 |


## How to read them, and what they do not show

* **Regression detection of 100% for the new gate is close to built in.** A `wrong_*` answer was written to
  fail the correctness checks (a test enforces it), so a gate that aggregates those checks will see it. What the
  table actually shows is that the aggregation and thresholds do not hide it, including when only part of the
  suite changes, and that a change confined to one evaluator group (`format_break`) is not diluted by the rest.
* **The old gate's misses are real design properties, but their size depends on my answers.** `wrong_close`
  answers are deliberately near the reference wording ("The capital of France is Lyon."), which is the case
  similarity cannot see; `wrong_far` answers are worded very differently, and there the old gate catches them
  (see the `wrong_far` rows). The v2 edge cases made the old gate look better than on v1 (91% vs 73% on the full
  suite) because many of their wrong answers are short or odd enough to differ a lot from the reference. A real
  model's wrong answers will fall somewhere in between; the table cannot say where.
* **The old gate's false blocks come from paraphrases that are correct but worded differently.** How often a
  real model does that is not measured here.
* **Small changes are not reliably caught.** The `single_wrong_close` row (exactly one case wrong) is blocked
  only when that case belongs to a small evaluator group, such as the 3-case `json_schema` group in the `pr`
  suite (one flip = 33%). In the 39-case `contains` group of the full suite one flip is ~2.6% and passes (0/10);
  in the 10-case `contains` group of the `pr` suite it is exactly 10%, which is not *more* than the threshold, so
  it passes too. The verdict is on aggregate scores per evaluator, so it is sensitive to group size and therefore
  to which suite you run. Cases that scored lower but stayed under the threshold are listed in `regressed_cases`
  and mentioned in the summary, but they do not block.
* The 10%-wrong rows are a sensitivity curve, not part of the headline: whether they *should* block is a policy
  choice.
* Latency and cost are held constant in the simulation, so they play no role here.
* Not measured: a real model's run-to-run variance (the gate has no noise margin yet), judge-based scoring,
  and anything about the CI workflow itself.

## Machine-readable output

`--out` writes the full JSON: per-scenario results (blocked or not under each gate, which evaluators flagged,
how many cases scored lower) for every simulated candidate.
