import logging
from Levenshtein import ratio as levenshtein_ratio
from rouge_score import rouge_scorer

logger = logging.getLogger(__name__)
_rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

def compute_rouge_l(prediction: str, reference: str) -> float:
    if not prediction.strip() or not reference.strip():
        return 0.0
    scores = _rouge.score(reference, prediction)
    return round(scores["rougeL"].fmeasure, 4)

def compute_levenshtein(prediction: str, reference: str) -> float:
    if not prediction.strip() or not reference.strip():
        return 0.0
    return round(levenshtein_ratio(prediction, reference), 4)

def compute_similarity(prediction: str, reference: str) -> float:
    rouge = compute_rouge_l(prediction, reference)
    lev = compute_levenshtein(prediction, reference)
    return round((rouge + lev) / 2, 4)