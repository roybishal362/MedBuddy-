"""
MedBuddy — Evaluation Metrics
ROUGE, BERTScore, hallucination rate, OCR accuracy, and custom metrics.
"""
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def compute_rouge(generated: str, reference: str) -> dict:
    """Compute ROUGE-1, ROUGE-2, ROUGE-L scores."""
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        scores = scorer.score(reference, generated)
        return {
            "rouge1": round(scores['rouge1'].fmeasure, 4),
            "rouge2": round(scores['rouge2'].fmeasure, 4),
            "rougeL": round(scores['rougeL'].fmeasure, 4)
        }
    except Exception as e:
        print(f"[Metrics] ROUGE error: {e}")
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}


def compute_bertscore(generated_list: list, reference_list: list) -> dict:
    """Compute BERTScore F1 across a list of generated/reference pairs."""
    try:
        from bert_score import score
        P, R, F1 = score(generated_list, reference_list, lang="en", verbose=False)
        return {
            "precision": round(P.mean().item(), 4),
            "recall": round(R.mean().item(), 4),
            "f1": round(F1.mean().item(), 4)
        }
    except Exception as e:
        print(f"[Metrics] BERTScore error: {e}")
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}


def compute_hallucination_rate(verification_results: list) -> dict:
    """
    Compute hallucination rate from a list of verification results.

    Args:
        verification_results: list of dicts with 'verdict' and 'grounding_score'

    Returns:
        dict with hallucination rate and mean grounding score
    """
    if not verification_results:
        return {"hallucination_rate": 0.0, "mean_grounding_score": 0.0, "std_grounding_score": 0.0}

    flagged = sum(1 for v in verification_results if v.get("verdict") == "FLAG")
    total = len(verification_results)

    scores = [v.get("grounding_score", 0) for v in verification_results if v.get("grounding_score") is not None]

    import numpy as np
    mean_score = np.mean(scores) if scores else 0.0
    std_score = np.std(scores) if scores else 0.0

    return {
        "hallucination_rate": round(flagged / total, 4),
        "mean_grounding_score": round(float(mean_score), 4),
        "std_grounding_score": round(float(std_score), 4),
        "total_verified": total,
        "total_flagged": flagged
    }


def compute_flag_accuracy(test_entities: list) -> dict:
    """
    Compute flag accuracy: % of NORMAL/BORDERLINE/CRITICAL flags correct.

    Args:
        test_entities: list of dicts with 'predicted_flag' and 'expected_flag'
    """
    if not test_entities:
        return {"flag_accuracy": 0.0, "total": 0}

    correct = sum(1 for e in test_entities if e.get("predicted_flag") == e.get("expected_flag"))
    total = len(test_entities)

    return {
        "flag_accuracy": round(correct / total, 4),
        "correct": correct,
        "total": total
    }


def compute_entity_extraction_accuracy(extracted: list, expected: list) -> dict:
    """
    Compute entity extraction accuracy.

    Args:
        extracted: list of extracted entity term names
        expected: list of expected entity term names
    """
    if not expected:
        return {"entity_accuracy": 0.0, "total_expected": 0}

    # Normalize terms for comparison
    extracted_normalized = {t.lower().strip() for t in extracted}
    expected_normalized = {t.lower().strip() for t in expected}

    matched = extracted_normalized & expected_normalized
    accuracy = len(matched) / len(expected_normalized) if expected_normalized else 0.0

    return {
        "entity_accuracy": round(accuracy, 4),
        "matched": len(matched),
        "total_expected": len(expected_normalized),
        "total_extracted": len(extracted_normalized),
        "missed": list(expected_normalized - extracted_normalized),
        "extra": list(extracted_normalized - expected_normalized)
    }


def compute_rag_tier_distribution(retrieval_results: list) -> dict:
    """
    Compute RAG tier distribution: % queries resolved at Tier 1 / 2 / 3.

    Args:
        retrieval_results: list of RAG retrieval result dicts with 'source' field
    """
    if not retrieval_results:
        return {"tier1": 0.0, "tier2": 0.0, "tier3": 0.0}

    counts = Counter()
    for r in retrieval_results:
        source = r.get("source", "LLM_fallback")
        if source == "FAISS":
            counts["tier1"] += 1
        elif source == "MedlinePlus_live":
            counts["tier2"] += 1
        else:
            counts["tier3"] += 1

    total = len(retrieval_results)
    return {
        "tier1_pct": round(counts["tier1"] / total * 100, 2),
        "tier2_pct": round(counts["tier2"] / total * 100, 2),
        "tier3_pct": round(counts["tier3"] / total * 100, 2),
        "tier1_count": counts["tier1"],
        "tier2_count": counts["tier2"],
        "tier3_count": counts["tier3"],
        "total": total
    }


def compute_ocr_accuracy(ocr_text: str, ground_truth: str) -> dict:
    """
    Compute OCR character error rate using jiwer.

    Args:
        ocr_text: OCR-extracted text
        ground_truth: Known correct text
    """
    try:
        from jiwer import cer
        error_rate = cer(ground_truth, ocr_text)
        return {
            "character_error_rate": round(error_rate, 4),
            "accuracy": round(1 - error_rate, 4)
        }
    except Exception as e:
        print(f"[Metrics] OCR accuracy error: {e}")
        return {"character_error_rate": 1.0, "accuracy": 0.0}
