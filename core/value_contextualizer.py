"""
MedBuddy — Value Contextualizer
Deterministic clinical risk flag assignment: NORMAL / BORDERLINE / CRITICAL / UNKNOWN
No LLM involved — pure Python logic.
"""


def contextualize(patient_value: str, ref_low: float, ref_high: float) -> str:
    """
    Assign clinical risk flag to an entity based on reference ranges.

    Returns: "NORMAL" | "BORDERLINE" | "CRITICAL" | "UNKNOWN"
    """
    try:
        val = float(patient_value)
    except (ValueError, TypeError):
        return "UNKNOWN"

    if ref_low is None or ref_high is None:
        return "UNKNOWN"

    try:
        ref_low = float(ref_low)
        ref_high = float(ref_high)
    except (ValueError, TypeError):
        return "UNKNOWN"

    if ref_low <= val <= ref_high:
        return "NORMAL"
    elif val < ref_low * 0.85 or val > ref_high * 1.15:
        return "CRITICAL"
    else:
        return "BORDERLINE"


def add_flags_to_entities(entities: list) -> list:
    """
    Add 'flag' field to each entity in the list.

    Args:
        entities: list of entity dicts from entity_extractor

    Returns:
        Same list with 'flag' field added to each entity
    """
    for entity in entities:
        flag = contextualize(
            entity.get("patient_value", ""),
            entity.get("reference_low"),
            entity.get("reference_high")
        )
        entity["flag"] = flag
    return entities
