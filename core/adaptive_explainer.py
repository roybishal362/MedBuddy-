"""
MedBuddy — Adaptive Explainer
Generates plain-language explanations using RAG context.
Handles all 3 RAG tiers with appropriate prompts.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config.settings import (
    primary_llm, LITERACY_LEVELS,
    RAG_EXPLANATION_PROMPT, LLM_FALLBACK_PROMPT,
    TERM_DICTIONARY_PROMPT, TERM_DICTIONARY_FALLBACK_PROMPT
)


def explain(
    term: str,
    value: str = "",
    unit: str = "",
    ref_range: str = "",
    flag: str = "",
    rag_result: dict = None,
    literacy_level: str = "Intermediate",
    language: str = "English"
) -> dict:
    """
    Generate a plain-language explanation for a medical term/result.

    Args:
        term: Medical term name
        value: Patient's value (optional, for Mode 2)
        unit: Unit of measurement
        ref_range: Reference range string
        flag: NORMAL/BORDERLINE/CRITICAL/UNKNOWN
        rag_result: Output from rag_retriever.retrieve()
        literacy_level: Basic / Intermediate / Advanced
        language: English / Hindi

    Returns:
      {
        "explanation": str,
        "source": str,
        "disclaimer": bool,
        "source_url": str | None
      }
    """
    literacy_desc = LITERACY_LEVELS.get(literacy_level, LITERACY_LEVELS["Intermediate"])
    rag_source = rag_result.get("source", "LLM_fallback") if rag_result else "LLM_fallback"
    disclaimer = rag_result.get("disclaimer", True) if rag_result else True
    context = rag_result.get("context") if rag_result else None
    source_url = rag_result.get("source_url") if rag_result else None

    try:
        if context and not disclaimer:
            # Tier 1 or Tier 2: RAG-grounded explanation
            prompt = RAG_EXPLANATION_PROMPT.format(
                retrieved_context=context,
                term=term,
                value=value if value else "N/A",
                unit=unit if unit else "",
                ref_range=ref_range if ref_range else "Not specified",
                flag=flag if flag else "N/A",
                literacy_level_description=literacy_desc,
                language=language
            )
        else:
            # Tier 3: LLM fallback
            prompt = LLM_FALLBACK_PROMPT.format(
                term=term,
                value=value if value else "N/A",
                unit=unit if unit else "",
                ref_range=ref_range if ref_range else "Not specified",
                flag=flag if flag else "N/A",
                literacy_level_description=literacy_desc,
                language=language
            )

        response = primary_llm.invoke(prompt)
        explanation = response.content.strip()

        return {
            "explanation": explanation,
            "source": rag_source,
            "disclaimer": disclaimer,
            "source_url": source_url
        }

    except Exception as e:
        print(f"[AdaptiveExplainer] Error: {e}")
        error_msg = {
            "English": f"Unable to generate explanation for {term}. Please try again.",
            "Hindi": f"{term} के लिए व्याख्या तैयार करने में असमर्थ। कृपया पुनः प्रयास करें।"
        }
        return {
            "explanation": error_msg.get(language, error_msg["English"]),
            "source": "error",
            "disclaimer": True,
            "source_url": None
        }


def explain_term(
    term: str,
    value: str = "",
    unit: str = "",
    rag_result: dict = None,
    literacy_level: str = "Intermediate",
    language: str = "English"
) -> dict:
    """
    Mode 1 (Ask a Term) explanation.

    - If the user did NOT provide a value -> dictionary-style answer:
      what the test is + the typical normal range (no patient-specific language).
    - If the user DID provide a value -> defer to the value-aware explain()
      so the number itself is interpreted.

    Returns the same dict shape as explain().
    """
    # Value supplied -> reuse the existing value-aware pipeline unchanged
    if value and str(value).strip():
        return explain(
            term=term, value=value, unit=unit,
            rag_result=rag_result, literacy_level=literacy_level, language=language
        )

    literacy_desc = LITERACY_LEVELS.get(literacy_level, LITERACY_LEVELS["Intermediate"])
    rag_source = rag_result.get("source", "LLM_fallback") if rag_result else "LLM_fallback"
    disclaimer = rag_result.get("disclaimer", True) if rag_result else True
    context = rag_result.get("context") if rag_result else None
    source_url = rag_result.get("source_url") if rag_result else None

    try:
        if context and not disclaimer:
            # Tier 1 / Tier 2: grounded dictionary explanation
            prompt = TERM_DICTIONARY_PROMPT.format(
                retrieved_context=context,
                term=term,
                literacy_level_description=literacy_desc,
                language=language
            )
        else:
            # Tier 3: dictionary explanation from general knowledge
            prompt = TERM_DICTIONARY_FALLBACK_PROMPT.format(
                term=term,
                literacy_level_description=literacy_desc,
                language=language
            )

        response = primary_llm.invoke(prompt)
        explanation = response.content.strip()

        return {
            "explanation": explanation,
            "source": rag_source,
            "disclaimer": disclaimer,
            "source_url": source_url
        }

    except Exception as e:
        print(f"[AdaptiveExplainer] explain_term Error: {e}")
        error_msg = {
            "English": f"Unable to generate explanation for {term}. Please try again.",
            "Hindi": f"{term} के लिए व्याख्या तैयार करने में असमर्थ। कृपया पुनः प्रयास करें।"
        }
        return {
            "explanation": error_msg.get(language, error_msg["English"]),
            "source": "error",
            "disclaimer": True,
            "source_url": None
        }
