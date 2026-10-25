# Versioned benchmarks

A benchmark is a named, versioned set of test cases with stable IDs. The shipped one is
`benchmarks/core-v2.yaml` (benchmark `core`, version `2`). Version 1 (`core-v1.yaml`, 39 cases) stays in the repo
unchanged, and v2 contains all of its cases verbatim (a test enforces it) plus 26 edge cases.

## File format

The YAML file is the API payload of `POST /datasets/import`:

```yaml
benchmark: core
version: "1"
cases:
  - case_key: geo-capital-france-001      # stable, lowercase, unique within the benchmark
    input_query: What is the capital of France?
    expected_output: The capital of France is Paris.
    task_type: qa                         # qa | classification | extraction | structured_output | instruction_following | summarization
    category: normal                      # normal | ambiguous | edge | long_input | long_output | empty_malformed | structured_output | similarity_unreliable
    difficulty: easy                      # easy | medium | hard
    suites: [pr, full]                    # every case is in 'full'; 'pr' is the fast subset
    eval_spec:                            # required: how the case is graded (see evaluators.md)
      - {name: contains, config: {all: [paris]}}
      - {name: text_similarity}
```

Invalid files are rejected with HTTP 422 and nothing is written (unknown evaluator, bad config, duplicate `case_key`,
unknown category, a case outside `full`, an empty `eval_spec`).

## Suites

| Suite | Cases | Use |
|---|---|---|
| `pr` | 16 | Runs on every pull request in CI. Small enough for a CPU-only runner, and covers every kind of check (`contains`, `exact_match`, `json_schema`, `json_equal`). |
| `full` | 65 | Everything. Run it on demand or on a schedule; not wired into CI. |

Categories in `full` (v2): normal (20), edge (21), structured_output (8), similarity_unreliable (5), ambiguous (3),
empty_malformed (3), long_input (3), long_output (2).

## What v2 adds (26 cases)

| Area | Cases |
|---|---|
| Empty and tiny input | whitespace-only message, a bare "Hi", emoji in the question |
| Other languages and scripts | Spanish, French, Japanese (matched without word boundaries; no similarity score, because the ROUGE tokenizer drops non-ASCII text) |
| Prompt injection inside data | a review that tells the model to answer "PWNED" (`pr`) |
| Number formats and arithmetic | thousands separators, large products, a unit calculation, a code snippet |
| Questions small models often get wrong | letters in "strawberry", reversing a word, "months with 28 days", the day after Saturday, an antonym |
| False premises and refusals | Great Wall visible from the Moon (`pr`), Einstein's Nobel Prize, a request for weapon instructions that must be refused |
| Ambiguity and sarcasm | a mixed-sentiment review (neutral or negative accepted), a sarcastic one |
| Long input with look-alike distractors | the right door code among near-identical codes |
| Structured-output corners | unicode in JSON, a `null` field (`pr`), an empty array (`pr`), a list of objects |

Some of these (the letter count, the injection, sarcasm) are hard for a small model, so a baseline may score 0 on them.
That is fine for comparing two runs, but read a case's absolute score with that in mind. None of the expected answers
has been checked against real `llama3.2` output.

## Loading

```bash
python scripts/load_benchmark.py benchmarks/core-v2.yaml --suite pr --api http://localhost:8000
# stderr: benchmark core v2: 65 created, 0 unchanged, suite 'pr' = 16 cases
# stdout: 1,2,3,...   (dataset ids, ready for POST /evaluations/run)
```

`GET /datasets/?benchmark=core&benchmark_version=2&suite=pr` lists the same cases (`category` also filters).

## Immutability

A case is identified by `(benchmark, version, case_key)`, and `content_hash` is a sha256 over its query, expected
output, `eval_spec`, task type, category, difficulty and suites. Re-importing an unchanged file is a no-op. Importing a
**changed** case under the same version is refused with **HTTP 409 and nothing is written**: scores from different days
must be scores on the same question. To change a case, publish a new `version`; runs compared with each other must
use the same cases anyway (the comparator checks case ids and evaluator versions).

## Honest limits of the core benchmark

* The expected answers and checks were written by hand and have **not** been validated against real `llama3.2` output.
  `tests/test_benchmark.py` checks that each reference answer passes its own checks and that an empty answer fails
  them, which catches typos, not unfair or unachievable cases.
* The checks are term- and structure-based. `contains` cannot see that "Paris is not the capital" negates the fact;
  cases where that matters use `none` terms, but it is a heuristic.
* 65 cases is a small sample: single-case differences move a group mean noticeably (see
  [gate-measurement.md](gate-measurement.md)). There is no noise margin for run-to-run model variance yet.
* Structured-output cases ask for JSON in the user message while `prompts/production.yaml` asks for "one short
  sentence"; how a given model resolves that conflict is untested here.
