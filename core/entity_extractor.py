"""
MedBuddy — Entity Extractor
Structured entity extraction from report text (Prompt B — background processing).
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config.settings import fallback_llm, ENTITY_EXTRACTOR_PROMPT
from core.utils import parse_json_response


def extract_entities(structured_report_text: str) -> list:
    """
    Extract structured entity list from report text.

    Returns list of dicts:
      [{
        "term": str,
        "abbreviation": str | None,
        "patient_value": str,
        "unit": str,
        "reference_range": str,
        "reference_low": float | None,
        "reference_high": float | None
      }]
    """
    prompt = ENTITY_EXTRACTOR_PROMPT.format(
        structured_report_text=structured_report_text
    )

    # Attempt 1
    try:
        response = fallback_llm.invoke(prompt)
        content = response.content.strip()

        entities = parse_json_response(content)
        if isinstance(entities, list):
            return _validate_entities(entities)

    except Exception as e:
        print(f"[EntityExtractor] Attempt 1 failed: {e}")

    # Attempt 2 with stricter prompt
    try:
        strict_prompt = (
            "CRITICAL: Return ONLY a JSON array. No text before or after. "
            "No markdown. No thinking. Start with [ and end with ].\n\n" + prompt
        )
        response = fallback_llm.invoke(strict_prompt)
        content = response.content.strip()

        entities = parse_json_response(content)
        if isinstance(entities, list):
            return _validate_entities(entities)

    except Exception as e:
        print(f"[EntityExtractor] Attempt 2 failed: {e}")

    # Both attempts failed — return empty list
    print("[EntityExtractor] WARNING: Both extraction attempts failed. Returning empty list.")
    return []


def _validate_entities(entities: list) -> list:
    """Validate and clean extracted entities."""
    validated = []
    for entity in entities:
        if not isinstance(entity, dict):
            continue

        cleaned = {
            "term": str(entity.get("term", "")).strip(),
            "abbreviation": entity.get("abbreviation"),
            "patient_value": str(entity.get("patient_value", "")).strip(),
            "unit": str(entity.get("unit", "")).strip() if entity.get("unit") else "",
            "reference_range": str(entity.get("reference_range", "")).strip() if entity.get("reference_range") else "",
            "reference_low": _safe_float(entity.get("reference_low")),
            "reference_high": _safe_float(entity.get("reference_high"))
        }

        # Only include if we have at least a term and value
        if cleaned["term"] and cleaned["patient_value"]:
            validated.append(cleaned)

    return validated


def _safe_float(value) -> float:
    """Safely convert to float or return None."""
    if value is None:
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None
