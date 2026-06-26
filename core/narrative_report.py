"""
MedBuddy — Narrative Report Handler
Handles NON-lab reports (radiology / imaging, ultrasound, X-ray, CT, MRI,
pathology / biopsy, discharge summaries, clinical notes).

Produces a plain-language summary + explained findings + a term glossary.
The numeric lab pipeline is untouched; this path runs only for NARRATIVE reports.
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config.settings import (
    fallback_llm, primary_llm, LITERACY_LEVELS,
    REPORT_KIND_PROMPT, NARRATIVE_REPORT_PROMPT
)
from core.utils import parse_json_response


def _heuristic_kind(report_text: str) -> str:
    """Offline fallback classifier when the LLM call is unavailable."""
    text = (report_text or "").lower()
    range_hits = len(re.findall(r"\d+\.?\d*\s*[-–]\s*\d+\.?\d*", report_text or ""))
    lab_kw = sum(k in text for k in (
        "reference range", "ref range", "normal range", "g/dl", "mg/dl",
        "mmol/l", "u/l", "iu/ml", "cells/", "/hpf"
    ))
    imaging_kw = sum(k in text for k in (
        "ultrasound", "sonograph", "x-ray", "x ray", "ct scan", "mri",
        "impression", "radiolog", "biopsy", "histopath", "scan", "doppler"
    ))
    if imaging_kw >= 1 and range_hits < 3:
        return "NARRATIVE"
    if range_hits >= 3 or lab_kw >= 2:
        return "LAB"
    return "NARRATIVE"


def classify_report_kind(report_text: str) -> str:
    """Return 'LAB' or 'NARRATIVE' for routing."""
    if not report_text or not report_text.strip():
        return "NARRATIVE"
    try:
        prompt = REPORT_KIND_PROMPT.format(report_text=report_text[:3000])
        resp = fallback_llm.invoke(prompt)
        out = parse_json_response(resp.content.strip())
        if out and isinstance(out, dict):
            kind = str(out.get("kind", "")).upper().strip()
            if kind in ("LAB", "NARRATIVE"):
                return kind
        return _heuristic_kind(report_text)
    except Exception as e:
        print(f"[NarrativeReport] classify error: {e}")
        return _heuristic_kind(report_text)


def analyze_narrative(report_text: str, literacy_level: str = "Intermediate",
                      language: str = "English") -> dict:
    """Produce {summary, findings, glossary} for a narrative report."""
    literacy_desc = LITERACY_LEVELS.get(literacy_level, LITERACY_LEVELS["Intermediate"])
    fallback = {
        "summary": (
            "We could read your report but could not fully structure it. "
            "Please review it together with your doctor."
            if language == "English" else
            "हम आपकी रिपोर्ट पढ़ सके लेकिन उसे पूरी तरह व्यवस्थित नहीं कर सके। "
            "कृपया इसे अपने डॉक्टर के साथ देखें।"
        ),
        "findings": [],
        "glossary": []
    }
    try:
        prompt = NARRATIVE_REPORT_PROMPT.format(
            report_text=report_text[:8000],
            literacy_level_description=literacy_desc,
            language=language
        )
        resp = primary_llm.invoke(prompt)
        out = parse_json_response(resp.content.strip())
        if not out or not isinstance(out, dict):
            return fallback

        summary = out.get("summary") or fallback["summary"]
        findings = out.get("findings") if isinstance(out.get("findings"), list) else []
        glossary = out.get("glossary") if isinstance(out.get("glossary"), list) else []

        clean_findings = [
            {
                "finding": str(f.get("finding", "")).strip(),
                "meaning": str(f.get("meaning", "")).strip(),
            }
            for f in findings if isinstance(f, dict) and f.get("finding")
        ]
        clean_glossary = [
            {
                "term": str(g.get("term", "")).strip(),
                "definition": str(g.get("definition", "")).strip(),
            }
            for g in glossary if isinstance(g, dict) and g.get("term")
        ]

        return {
            "summary": str(summary).strip(),
            "findings": clean_findings,
            "glossary": clean_glossary,
        }
    except Exception as e:
        print(f"[NarrativeReport] analyze error: {e}")
        return fallback
