"""
MedBuddy — Intent Classifier
Classifies Mode 1 queries: MEDICAL_TERM / REPORT_QUESTION / OUT_OF_SCOPE
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config.settings import fallback_llm, INTENT_CLASSIFIER_PROMPT, REFUSAL_KEYWORDS
from core.utils import parse_json_response


def classify(query: str) -> dict:
    """
    Classify user query intent.

    Returns:
      {
        "intent": "MEDICAL_TERM" | "REPORT_QUESTION" | "OUT_OF_SCOPE",
        "confidence": float,
        "normalized_term": str | None
      }
    """
    # Quick keyword check for obvious refusal cases
    query_lower = query.lower().strip()
    for keyword in REFUSAL_KEYWORDS:
        if keyword in query_lower:
            return {
                "intent": "OUT_OF_SCOPE",
                "confidence": 0.95,
                "normalized_term": None
            }

    # LLM-based classification
    try:
        prompt = INTENT_CLASSIFIER_PROMPT.format(query=query)
        response = fallback_llm.invoke(prompt)
        content = response.content.strip()

        result = parse_json_response(content)

        if result and isinstance(result, dict):
            intent = result.get("intent", "OUT_OF_SCOPE")
            confidence = float(result.get("confidence", 0.0))
            normalized_term = result.get("normalized_term")

            # Confidence gate: below 0.6 for MEDICAL_TERM → treat as OUT_OF_SCOPE
            if intent == "MEDICAL_TERM" and confidence < 0.6:
                return {
                    "intent": "OUT_OF_SCOPE",
                    "confidence": confidence,
                    "normalized_term": None
                }

            return {
                "intent": intent,
                "confidence": confidence,
                "normalized_term": normalized_term
            }
        else:
            raise ValueError("Failed to parse JSON from LLM response")

    except Exception as e:
        print(f"[IntentClassifier] Error: {e}")
        # On error, default to treating as medical term if short query
        if len(query.split()) <= 4:
            return {
                "intent": "MEDICAL_TERM",
                "confidence": 0.5,
                "normalized_term": query.strip()
            }
        return {
            "intent": "OUT_OF_SCOPE",
            "confidence": 0.5,
            "normalized_term": None
        }
