from app.core.evaluators.base import EvalInput, EvalResult, Evaluator, EvaluatorKind
from app.core.evaluators.registry import register
from app.core.metrics import compute_levenshtein, compute_rouge_l


@register
class TextSimilarity(Evaluator):
    """
    Mean of ROUGE-L F-measure and Levenshtein ratio (the original PromptGuard metric,
    unchanged: same formula, same rounding).

    This measures how alike two strings LOOK. A fluent paraphrase scores low and a
    wrong answer that reuses the reference's wording scores high, so it is an
    indicator and not a correctness check.
    """

    name = "text_similarity"
    version = "1"
    kind = EvaluatorKind.SIMILARITY

    def evaluate(self, inp: EvalInput) -> EvalResult:
        rouge = compute_rouge_l(inp.prediction, inp.reference)
        lev = compute_levenshtein(inp.prediction, inp.reference)
        return EvalResult(
            score=round((rouge + lev) / 2, 4),
            details={"rouge_l": rouge, "levenshtein": lev},   # raw sub-scores
        )
