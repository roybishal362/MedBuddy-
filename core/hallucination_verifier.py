"""
MedBuddy — Hallucination Verifier
Self-critique grounding check for generated explanations.
Applies to Tier 1 and Tier 2 only. Tier 3 is already labelled unverified.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config.settings import fallback_llm, HALLUCINATION_VERIFICATION_PROMPT
from core.utils import parse_json_response


def verify(context: str, explanation: str) -> dict:
    """
    Verify a generated explanation is grounded in its retrieved context.

    Args:
        context: The RAG-retrieved context text
        explanation: The generated explanation text

    Returns:
      {
        "grounding_score": float (0.0 - 1.0),
        "ungrounded_claims": list,
        "verdict": "PASS" | "FLAG"
      }
    """
    if not context or not explanation:
        return {
            "grounding_score": None,
            "ungrounded_claims": [],
            "verdict": "FLAG"
        }

    try:
        prompt = HALLUCINATION_VERIFICATION_PROMPT.format(
            context=context,
            explanation=explanation
        )
        response = fallback_llm.invoke(prompt)
        content = response.content.strip()

        result = parse_json_response(content)

        if result and isinstance(result, dict):
            grounding_score = float(result.get("grounding_score", 0.0))
            ungrounded = result.get("ungrounded_claims", [])
            verdict = "PASS" if grounding_score >= 0.85 else "FLAG"

            return {
                "grounding_score": round(grounding_score, 4),
                "ungrounded_claims": ungrounded if isinstance(ungrounded, list) else [],
                "verdict": verdict
            }
        else:
            raise ValueError("Failed to parse verification JSON")

    except Exception as e:
        print(f"[HallucinationVerifier] Error: {e}")
        return {
            "grounding_score": 0.5,
            "ungrounded_claims": ["Unable to verify — parse error"],
            "verdict": "FLAG"
        }
