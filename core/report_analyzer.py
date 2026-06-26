"""
MedBuddy — Report Analyzer
Holistic LLM comprehension of the full structured report text.
Produces an internal representation used by summarizer and extractor.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config.settings import primary_llm, REPORT_ANALYZER_PROMPT
from core.utils import parse_json_response


def analyze(structured_report_text: str) -> dict:
    """
    LLM reads the full structured report text holistically.

    Returns:
      {
        "overall_assessment": str,
        "critical_findings": list,
        "normal_findings": list,
        "report_type": str,
        "patient_indicators": dict
      }
    """
    try:
        prompt = REPORT_ANALYZER_PROMPT.format(
            structured_report_text=structured_report_text
        )
        response = primary_llm.invoke(prompt)
        content = response.content.strip()

        result = parse_json_response(content)

        if result and isinstance(result, dict):
            return {
                "overall_assessment": result.get("overall_assessment", "Analysis completed"),
                "critical_findings": result.get("critical_findings", []),
                "normal_findings": result.get("normal_findings", []),
                "report_type": result.get("report_type", "Lab Report"),
                "patient_indicators": result.get("patient_indicators", {})
            }
        else:
            raise ValueError("Failed to parse JSON from LLM response")

    except Exception as e:
        print(f"[ReportAnalyzer] Error parsing LLM response: {e}")
        return {
            "overall_assessment": "Report analysis completed. Please review detailed results below.",
            "critical_findings": [],
            "normal_findings": [],
            "report_type": "Lab Report",
            "patient_indicators": {}
        }
