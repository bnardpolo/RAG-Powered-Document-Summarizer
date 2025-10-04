from __future__ import annotations
from typing import List, Tuple, Set, Optional, Dict, Any
import re
import numpy as np

# ---------------- Retrieval metrics (model-agnostic) ----------------
def precision_at_k(ranked_ids: List[int], relevant_ids: Set[int], k: int) -> float:
    cut = ranked_ids[:k]
    hits = sum(1 for x in cut if x in relevant_ids)
    return hits / max(k, 1)

def recall_at_k(ranked_ids: List[int], relevant_ids: Set[int], k: int) -> float:
    if not relevant_ids:
        return 0.0
    cut = ranked_ids[:k]
    hits = sum(1 for x in cut if x in relevant_ids)
    return hits / len(relevant_ids)

def mrr(ranked_ids: List[int], relevant_ids: Set[int]) -> float:
    for i, rid in enumerate(ranked_ids, start=1):
        if rid in relevant_ids:
            return 1.0 / i
    return 0.0

def ndcg_at_k(ranked_ids: List[int], relevant_ids: Set[int], k: int) -> float:
    gains = [1.0 if rid in relevant_ids else 0.0 for rid in ranked_ids[:k]]
    def dcg(gs):  # log2(i+2) so first rank uses log2(2)=1
        return sum(g / np.log2(i + 2) for i, g in enumerate(gs))
    ideal = sorted(gains, reverse=True)
    denom = dcg(ideal)
    if denom == 0:
        return 0.0
    return dcg(gains) / denom

# ---------------- Summarization metrics ----------------
def rouge_l(candidate: str, reference: str) -> float:
    """
    ROUGE-L F1 over tokens using LCS.
    """
    def lcs(a: List[str], b: List[str]) -> int:
        n, m = len(a), len(b)
        dp = [[0] * (m + 1) for _ in range(n + 1)]
        for i in range(n):
            ai = a[i]
            dpi1 = dp[i + 1]
            dpi = dp[i]
            for j in range(m):
                dpi1[j + 1] = dpi[j] + 1 if ai == b[j] else max(dpi[j + 1], dpi1[j])
        return dp[n][m]

    a = candidate.split()
    b = reference.split()
    if not a or not b:
        return 0.0
    L = lcs(a, b)
    prec = L / len(a)
    rec  = L / len(b)
    return 0.0 if (prec + rec) == 0 else (2 * prec * rec) / (prec + rec)

def bertscore_optional(candidate: str, reference: str) -> Optional[Dict[str, float]]:
    """
    Returns {'bertscore_p','bertscore_r','bertscore_f1'} or None if bert-score isn't installed.
    """
    try:
        from bert_score import score
        P, R, F1 = score([candidate], [reference], lang="en", rescale_with_baseline=True)
        return {
            "bertscore_p": float(P.mean()),
            "bertscore_r": float(R.mean()),
            "bertscore_f1": float(F1.mean()),
        }
    except Exception:
        return None

# ---------------- Hallucination / obscurity check ----------------
def entity_hallucination_report(summary: str, source: str) -> Dict[str, Any]:
    """
    Flags named-like entities that appear in summary but not in source.
    Tries spaCy NER; falls back to a simple ProperNoun regex.
    Always returns 'entity_coverage_rate' when possible.
    """
    # Try spaCy first
    try:
        import spacy
        try:
            nlp = spacy.load("en_core_web_sm")
            sum_ents = {e.text.strip() for e in nlp(summary or "").ents if e.text.strip()}
            src_ents = {e.text.strip() for e in nlp(source or "").ents if e.text.strip()}
            if sum_ents:
                coverage = len(sum_ents & src_ents) / len(sum_ents)
            else:
                coverage = 1.0  # no entities claimed -> no hallucination risk
            return {
                "summary_entities": sorted(sum_ents),
                "missing_in_source": sorted(list(sum_ents - src_ents)),
                "entity_coverage_rate": coverage,
                "backend": "spacy",
            }
        except OSError:
            # spaCy installed but model missing; fall through to regex
            pass
    except Exception:
        # spaCy not installed; fall through to regex
        pass

    # Regex fallback: capitalized tokens of length >= 3 (very approximate)
    cap = re.compile(r"\b[A-Z][a-zA-Z0-9_-]{2,}\b")
    sum_ents = set(cap.findall(summary or ""))
    src_ents = set(cap.findall(source or ""))
    if sum_ents:
        coverage = len(sum_ents & src_ents) / len(sum_ents)
    else:
        coverage = 1.0
    return {
        "summary_entities": sorted(sum_ents),
        "missing_in_source": sorted(list(sum_ents - src_ents)),
        "entity_coverage_rate": coverage,
        "backend": "regex",
    }
