"""
MedBuddy — Report Summarizer
Generates narrative summary shown to user immediately after upload (Prompt A).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config.settings import primary_llm, REPORT_SUMMARIZER_PROMPT, LITERACY_LEVELS


def summarize(
    structured_report_text: str,
    report_analyzer_output: dict,
    literacy_level: str = "Intermediate",
    language: str = "English"
) -> str:
    """
    Generate narrative summary of a medical report.

    Args:
        structured_report_text: Full extracted text from document intelligence
        report_analyzer_output: Output from report_analyzer.analyze()
        literacy_level: Basic / Intermediate / Advanced
        language: English / Hindi

    Returns:
        str: Narrative summary text
    """
    import json

    literacy_desc = LITERACY_LEVELS.get(literacy_level, LITERACY_LEVELS["Intermediate"])

    # Format analyzer output as readable string
    analyzer_str = json.dumps(report_analyzer_output, indent=2, ensure_ascii=False)

    try:
        prompt = REPORT_SUMMARIZER_PROMPT.format(
            structured_report_text=structured_report_text,
            report_analyzer_output=analyzer_str,
            literacy_level=literacy_level,
            literacy_level_description=literacy_desc,
            language=language
        )
        response = primary_llm.invoke(prompt)
        return response.content.strip()

    except Exception as e:
        print(f"[ReportSummarizer] Error: {e}")
        if language == "Hindi":
            return "रिपोर्ट का सारांश तैयार करने में एक त्रुटि हुई। कृपया पुनः प्रयास करें।"
        return "An error occurred while generating the report summary. Please try again."
