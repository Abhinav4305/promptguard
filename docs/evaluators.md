# Evaluators

An evaluator turns `(model output, reference answer, case metadata)` into **one score per test case**. It knows
nothing about Celery, the database or run comparison. The executor runs the evaluators of a case and stores one row
per evaluator in `evaluation_scores`; the comparator aggregates those rows.

## Contract

Every evaluator must (enforced when it is registered, `app/core/evaluators/base.py`):

| Property | Rule |
|---|---|
| `name`, `version` | Stored with every score. Bump `version` whenever scoring behaviour changes. |
| `kind` | `correctness` (objective, task-specific check) or `similarity` (surface text similarity) |
| score | A finite number in `[0, 1]`, higher is better (`direction = higher_is_better`) |
| aggregation | Arithmetic mean over cases |
| config | JSON object, validated at construction; its hash (`config_hash`) is stored with the score |

An evaluator that raises, or returns NaN / infinity / out-of-range / a non-number, produces an explicit `ERROR`
score row; the case is marked `FAILED` with `error_category = EVALUATOR_ERROR`. Scores are never silently dropped,
and **aggregation refuses to average over a missing or invalid score** (`MissingScoreError`).

## Built-in evaluators

| Name | Kind | Scores 1.0 when | Config |
|---|---|---|---|
| `text_similarity` (v1) | similarity | n/a, returns `round((ROUGE-L + Levenshtein ratio) / 2, 4)`, the original metric | none |
| `exact_match` | correctness | the normalised output equals the reference or an alternative | `case_sensitive` (true), `trim` (true), `collapse_whitespace`, `ignore_punctuation`, `alternatives` |
| `contains` | correctness | output has every `all` term, one `any` term, no `none` term | `all`, `any`, `none`, `case_sensitive` (false), `whole_word` (true) |
| `json_schema` | correctness | output parses as JSON and validates against a JSON Schema (draft 2020-12) | `schema` (required), `lenient` |
| `json_equal` | correctness | fraction of reference keys whose values match (partial credit); non-objects: equality | `fields`, `ignore_keys`, `number_tolerance`, `strict_keys`, `lenient` |

`lenient` accepts exactly one wrapper models add constantly, a fenced ```` ```json ```` block. Nothing else is repaired.

**Why `text_similarity` is not enough on its own.** "The capital of France is Lyon." scores higher against
"The capital of France is Paris." than the correct "Paris." does (a test pins this), and a correct answer in different
words can look like a large regression. Similarity is kept as an indicator, not as the thing that decides.

## Which scores can block a release (gating)

A case lists its evaluators in `eval_spec`:

```json
[{"name": "contains", "config": {"all": ["paris"]}}, {"name": "text_similarity"}]
```

* A **correctness** evaluator gates by default.
* A **similarity** evaluator gates only if the case has no correctness evaluator; next to one it is advisory.
* An entry can override with `"gate": true | false`.
* A case with no `eval_spec` is scored by `text_similarity` alone and gates on it, exactly as before.

The decision is stored on every score row (`gating`), so history stays interpretable if the rule changes later.

## How runs are compared

Per evaluator (name + version + gating), over the cases it scored, the comparator computes the mean for baseline and
candidate. A gating evaluator **fails the verdict** when its mean drops by more than its threshold relative to the
baseline: `REGRESSION_SIMILARITY_DROP_THRESHOLD` for similarity evaluators, `REGRESSION_CORRECTNESS_DROP_THRESHOLD`
for correctness evaluators (both 0.10 by default). Non-gating drops are reported as `advisory_regression`.

Before any of that, the runs must have been scored by the **same evaluators**: per case, the set of
`(name, version, config_hash, gating)` must match, else the comparison is `INCOMPARABLE` (`EVALUATOR_MISMATCH`).
A succeeded case with a missing, `ERROR` or invalid score is `INCOMPLETE` (`MISSING_SCORES`), and a case with no
gating evaluator at all is `INCOMPLETE` (`NO_GATING_SCORE`), because it could never block anything.

Known limit: the verdict is on aggregates, so a single flipped case in a large evaluator group stays under the
threshold. Such cases are listed in `regressed_cases` and mentioned in the summary but do not block. See
[gate-measurement.md](gate-measurement.md).

## Adding an evaluator

```python
from app.core.evaluators.base import EvalInput, EvalResult, Evaluator, EvaluatorKind
from app.core.evaluators.registry import register

@register
class StartsWith(Evaluator):
    name = "starts_with"
    version = "1"
    kind = EvaluatorKind.CORRECTNESS

    def evaluate(self, inp: EvalInput) -> EvalResult:
        return EvalResult(1.0 if inp.prediction.startswith(self.config["prefix"]) else 0.0)
```

Import the module in `app/core/evaluators/__init__.py`. No migration, executor or comparator change is needed: scores
live in rows, not columns. Add tests next to `tests/test_evaluators.py`.

## Outline: a future LLM judge, and how to calibrate it before it may gate

Not implemented. The interface already fits it: `kind = correctness`, `config` carrying the rubric, judge model and
prompt version, `details` carrying the rationale. It would be added as a new evaluator (for example `llm_judge`, v1),
**non-gating at first** (`"gate": false`), so it can run alongside the objective checks without affecting verdicts.

Before it is allowed to gate, calibrate it:

1. **Human labels.** Have people grade a few hundred outputs (from real runs, including ones the objective checks
   disagree on) as acceptable / not acceptable against the same rubric. Record who labelled and the guidelines; measure
   inter-annotator agreement, because the judge cannot beat it.
2. **Agreement.** Run the judge on the same set; report agreement and Cohen's kappa with the human majority, plus the
   false-accept rate (judge passes what humans reject), which matters most for a release gate. Split by category.
3. **Stability.** Repeat the judge several times at the intended temperature; report score variance. Set the
   comparator's threshold above that noise.
4. **Known biases.** Check position, verbosity and self-preference bias (a judge from the same family as the candidate)
   with swapped-order and length-controlled variants.
5. **Pin everything.** Judge model id, prompt, rubric and temperature go in `config`, so a change produces a new
   `config_hash` and the comparator refuses to mix old and new scores.
6. **Promote deliberately.** Only after the false-accept rate and stability meet a threshold agreed in advance does the
   evaluator become gating, and only for the categories where it was calibrated. Keep a held-out human-labelled set and
   re-measure whenever the judge model or rubric changes.
