import math

import pytest

from app.core.evaluators import (
    EvalInput, EvalResult, Evaluator, EvaluatorConfigError, EvaluatorKind, MissingScoreError,
    OutcomeStatus, aggregate, available, create_evaluator, resolve_evaluators, run_evaluator,
)
from app.core.evaluators.base import config_hash
from app.core.evaluators.registry import register
from app.core.metrics import compute_similarity


def score(name, prediction, reference="", **config):
    outcome = run_evaluator(create_evaluator(name, config), EvalInput(prediction, reference))
    assert outcome.status == OutcomeStatus.OK, outcome.error_message
    return outcome.score


# ── registry and contract ───────────────────────────────────────────────────────────

def test_builtin_evaluators_are_registered():
    assert available() == ["contains", "exact_match", "json_equal", "json_schema", "text_similarity"]


def test_unknown_evaluator_is_a_config_error_listing_the_options():
    with pytest.raises(EvaluatorConfigError, match="available"):
        create_evaluator("vibes")


def test_registration_rejects_evaluators_the_comparator_cannot_handle():
    class LowerIsBetter(Evaluator):
        name, version, kind, direction = "bad_direction", "1", EvaluatorKind.CORRECTNESS, "lower_is_better"
        def evaluate(self, inp): return EvalResult(0.0)

    class WrongRange(Evaluator):
        name, version, kind, value_range = "bad_range", "1", EvaluatorKind.CORRECTNESS, (0.0, 10.0)
        def evaluate(self, inp): return EvalResult(0.0)

    with pytest.raises(TypeError, match="direction"):
        register(LowerIsBetter)
    with pytest.raises(TypeError, match="value_range"):
        register(WrongRange)


def test_descriptor_carries_everything_needed_to_interpret_a_stored_score():
    d = create_evaluator("exact_match", {"case_sensitive": False}).descriptor()
    assert d["name"] == "exact_match" and d["version"] == "1" and d["kind"] == "correctness"
    assert d["direction"] == "higher_is_better" and d["aggregation"] == "mean" and d["value_range"] == [0.0, 1.0]
    assert d["config"] == {"case_sensitive": False} and len(d["config_hash"]) == 16


def test_config_hash_ignores_key_order_and_changes_with_content():
    assert config_hash({"a": 1, "b": [1, 2]}) == config_hash({"b": [1, 2], "a": 1})
    assert config_hash({"a": 1}) != config_hash({"a": 2})


# ── text_similarity ─────────────────────────────────────────────────────────────────

def test_text_similarity_reproduces_the_original_metric_exactly():
    for pred, ref in [("The capital of France is Paris.", "The capital of France is Paris."),
                      ("Paris", "The capital of France is Paris."), ("blue sky", "The sky is blue.")]:
        assert score("text_similarity", pred, ref) == compute_similarity(pred, ref)


def test_text_similarity_records_raw_sub_scores():
    outcome = run_evaluator(create_evaluator("text_similarity"), EvalInput("Paris", "The capital of France is Paris."))
    assert set(outcome.details) == {"rouge_l", "levenshtein"}


def test_surface_similarity_can_rank_a_wrong_answer_above_a_correct_one():
    """The reason correctness evaluators exist: text similarity is not correctness."""
    reference = "The capital of France is Paris."
    wrong = "The capital of France is Lyon."
    correct_paraphrase = "Paris."
    assert score("text_similarity", wrong, reference) > score("text_similarity", correct_paraphrase, reference)
    # ...while the correctness check gets both right:
    assert score("contains", wrong, all=["paris"]) == 0.0
    assert score("contains", correct_paraphrase, all=["paris"]) == 1.0


# ── exact_match ─────────────────────────────────────────────────────────────────────

def test_exact_match_is_strict_by_default_but_ignores_surrounding_whitespace():
    assert score("exact_match", "  4\n", "4") == 1.0
    assert score("exact_match", "Four", "four") == 0.0
    assert score("exact_match", "4.", "4") == 0.0


def test_exact_match_options():
    assert score("exact_match", "Four", "four", case_sensitive=False) == 1.0
    assert score("exact_match", "4.", "4", ignore_punctuation=True) == 1.0
    assert score("exact_match", "a   b", "a b", collapse_whitespace=True) == 1.0
    assert score("exact_match", "a   b", "a b") == 0.0


def test_exact_match_alternatives_and_details():
    outcome = run_evaluator(create_evaluator("exact_match", {"alternatives": ["4", "four"]}),
                            EvalInput("four", "4"))
    assert outcome.score == 1.0 and outcome.details["matched"] == "alternative[1]"
    miss = run_evaluator(create_evaluator("exact_match"), EvalInput("5", "4"))
    assert miss.score == 0.0 and miss.details["matched"] is None


@pytest.mark.parametrize("cfg", [{"nope": 1}, {"alternatives": "4"}, {"alternatives": [4]}])
def test_exact_match_rejects_bad_config(cfg):
    with pytest.raises(EvaluatorConfigError):
        create_evaluator("exact_match", cfg)


# ── contains ────────────────────────────────────────────────────────────────────────

def test_contains_accepts_any_phrasing_that_has_the_facts():
    for answer in ["Paris.", "It is Paris!", "France's capital city is paris"]:
        assert score("contains", answer, all=["Paris"]) == 1.0
    assert score("contains", "It is Lyon.", all=["Paris"]) == 0.0


def test_contains_all_any_none():
    cfg = dict(all=["366"], any=["leap", "february"], none=["365 days"])
    assert score("contains", "A leap year has 366 days.", **cfg) == 1.0
    assert score("contains", "It has 366 days.", **cfg) == 0.0                       # no 'any' term
    assert score("contains", "A leap year has 366 days, not 365 days.", **cfg) == 0.0  # forbidden term


def test_contains_whole_word_prevents_false_positives():
    assert score("contains", "There are 14 apples.", all=["4"]) == 0.0
    assert score("contains", "There are 14 apples.", all=["4"], whole_word=False) == 1.0


def test_contains_case_sensitivity_and_details():
    assert score("contains", "paris", all=["Paris"], case_sensitive=True) == 0.0
    out = run_evaluator(create_evaluator("contains", {"all": ["a", "b"], "none": ["z"]}), EvalInput("a z", ""))
    assert out.details == {"missing_required": ["b"], "any_matched": True, "forbidden_found": ["z"]}


@pytest.mark.parametrize("cfg", [{}, {"all": []}, {"all": [""]}, {"all": "x"}, {"all": ["x"], "bogus": 1}])
def test_contains_rejects_bad_config(cfg):
    with pytest.raises(EvaluatorConfigError):
        create_evaluator("contains", cfg)


# ── json_schema ─────────────────────────────────────────────────────────────────────

PERSON = {"type": "object", "required": ["name", "age"],
          "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
          "additionalProperties": False}


def test_json_schema_validates_shape():
    assert score("json_schema", '{"name": "Ada", "age": 36}', schema=PERSON) == 1.0
    assert score("json_schema", '{"name": "Ada", "age": "36"}', schema=PERSON) == 0.0
    assert score("json_schema", '{"name": "Ada"}', schema=PERSON) == 0.0
    assert score("json_schema", '{"name": "Ada", "age": 36, "x": 1}', schema=PERSON) == 0.0


def test_json_schema_distinguishes_parse_failures_from_shape_failures():
    parse = run_evaluator(create_evaluator("json_schema", {"schema": PERSON}), EvalInput("Sure! Here you go: {", ""))
    assert parse.score == 0.0 and parse.details["stage"] == "parse"
    shape = run_evaluator(create_evaluator("json_schema", {"schema": PERSON}), EvalInput('{"name": 1, "age": 2}', ""))
    assert shape.details["stage"] == "validate" and "name" in shape.details["errors"][0]


def test_json_schema_accepts_a_fenced_block_only_when_lenient():
    fenced = '```json\n{"name": "Ada", "age": 36}\n```'
    assert score("json_schema", fenced, schema=PERSON) == 1.0
    assert score("json_schema", fenced, schema=PERSON, lenient=False) == 0.0
    assert score("json_schema", 'Here: {"name": "Ada", "age": 36}', schema=PERSON) == 0.0   # no other repair


@pytest.mark.parametrize("cfg", [{}, {"schema": "nope"}, {"schema": {"type": "wat"}}, {"schema": {}, "x": 1}])
def test_json_schema_rejects_bad_config(cfg):
    with pytest.raises(EvaluatorConfigError):
        create_evaluator("json_schema", cfg)


# ── json_equal ──────────────────────────────────────────────────────────────────────

REF = '{"name": "Ada", "age": 36, "langs": ["en", "fr"], "active": true}'


def test_json_equal_ignores_key_order_and_whitespace():
    assert score("json_equal", '{"active":true,"langs":["en","fr"],"age":36,"name":"Ada"}', REF) == 1.0


def test_json_equal_gives_partial_credit_per_key_and_names_the_wrong_ones():
    out = run_evaluator(create_evaluator("json_equal"), EvalInput(
        '{"name": "Ada", "age": 37, "langs": ["en"], "active": true}', REF))
    assert out.score == 0.5 and out.details["wrong_keys"] == ["age", "langs"]


def test_json_equal_does_not_confuse_booleans_and_numbers():
    assert score("json_equal", '{"a": 1}', '{"a": true}') == 0.0
    assert score("json_equal", '{"a": true}', '{"a": true}') == 1.0


def test_json_equal_options():
    assert score("json_equal", '{"price": 9.99}', '{"price": 10}', number_tolerance=0.05) == 1.0
    assert score("json_equal", '{"price": 9.99}', '{"price": 10}') == 0.0
    assert score("json_equal", '{"a": 1, "b": 99}', '{"a": 1, "b": 2}', fields=["a"]) == 1.0
    assert score("json_equal", '{"a": 1, "ts": "now"}', '{"a": 1, "ts": "then"}', ignore_keys=["ts"]) == 1.0
    assert score("json_equal", '{"a": 1, "extra": 1}', '{"a": 1}') == 1.0
    assert score("json_equal", '{"a": 1, "extra": 1}', '{"a": 1}', strict_keys=True) == 0.0


def test_json_equal_scores_unparseable_output_zero_but_a_broken_reference_is_an_error():
    assert score("json_equal", "not json", REF) == 0.0
    broken = run_evaluator(create_evaluator("json_equal"), EvalInput('{"a": 1}', "not json"))
    assert broken.status == OutcomeStatus.ERROR and "reference is not valid JSON" in broken.error_message


def test_json_equal_non_object_reference_and_nested_values():
    assert score("json_equal", "[1, 2, 3]", "[1,2,3]") == 1.0
    assert score("json_equal", "[1, 3, 2]", "[1,2,3]") == 0.0
    assert score("json_equal", '{"a": {"b": [1, {"c": 2}]}}', '{"a": {"b": [1, {"c": 2}]}}') == 1.0


# ── run_evaluator normalises every failure into an explicit ERROR ─────────────────────

@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -0.1, 1.01, None, "0.5", True])
def test_invalid_scores_become_errors_not_numbers(bad):
    class Broken(Evaluator):
        name, version, kind = "broken_for_test", "1", EvaluatorKind.CORRECTNESS
        def evaluate(self, inp): return EvalResult(bad)

    out = run_evaluator(Broken(), EvalInput("x", "y"))
    assert out.status == OutcomeStatus.ERROR and out.score is None and "invalid score" in out.error_message


def test_evaluator_exceptions_are_captured():
    class Boom(Evaluator):
        name, version, kind = "boom_for_test", "1", EvaluatorKind.CORRECTNESS
        def evaluate(self, inp): raise RuntimeError("kaput")

    out = run_evaluator(Boom(), EvalInput("x", "y"))
    assert out.status == OutcomeStatus.ERROR and out.error_message == "RuntimeError: kaput"


# ── aggregation never skips missing scores ──────────────────────────────────────────────

def test_aggregate_is_the_mean_when_every_score_is_valid():
    assert aggregate([1.0, 0.5, 0.0]) == 0.5


@pytest.mark.parametrize("scores", [[], [1.0, None], [1.0, math.nan], [1.0, 1.5], [0.2, "x"]])
def test_aggregate_refuses_missing_or_invalid_scores(scores):
    with pytest.raises(MissingScoreError):
        aggregate(scores)


# ── spec resolution and gating ──────────────────────────────────────────────────────────

def test_no_spec_means_the_original_behaviour():
    (only,) = resolve_evaluators(None)
    assert only.evaluator.name == "text_similarity" and only.gating is True
    assert [r.evaluator.name for r in resolve_evaluators([])] == ["text_similarity"]


def test_similarity_is_advisory_next_to_a_correctness_check():
    resolved = {r.evaluator.name: r.gating for r in resolve_evaluators(
        [{"name": "contains", "config": {"all": ["x"]}}, {"name": "text_similarity"}])}
    assert resolved == {"contains": True, "text_similarity": False}


def test_gate_can_be_overridden_per_entry():
    resolved = {r.evaluator.name: r.gating for r in resolve_evaluators(
        [{"name": "contains", "config": {"all": ["x"]}, "gate": False},
         {"name": "text_similarity", "gate": True}])}
    assert resolved == {"contains": False, "text_similarity": True}


@pytest.mark.parametrize("spec", [
    [{"name": "text_similarity"}, {"name": "text_similarity"}],
    [{"config": {}}],
    [{"name": "text_similarity", "typo": 1}],
    [{"name": "nope"}],
    [{"name": "contains"}],
])
def test_bad_specs_are_rejected(spec):
    with pytest.raises(EvaluatorConfigError):
        resolve_evaluators(spec)
