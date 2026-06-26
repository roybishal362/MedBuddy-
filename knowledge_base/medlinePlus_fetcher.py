"""
MedBuddy — MedlinePlus Fetcher
Two functions: build_fetch (offline KB build) and live_fetch (runtime Tier 2).
"""
import os
import json
import re
import requests
import time

# Config
MEDLINEPLUS_API_BASE = "https://connect.medlineplus.gov/application"
KB_RAW_DIR = os.path.join(os.path.dirname(__file__), "raw")


def _fetch_term(term: str) -> dict:
    """Fetch a single term definition from MedlinePlus Connect API."""
    try:
        params = {
            "mainSearchCriteria.v.c": term,
            "knowledgeResponseType": "application/json"
        }
        response = requests.get(MEDLINEPLUS_API_BASE, params=params, timeout=15)

        if response.status_code != 200:
            return None

        data = response.json()

        # Parse the MedlinePlus response
        entries = data.get("feed", {}).get("entry", [])
        if not entries:
            return None

        entry = entries[0]
        title = entry.get("title", {}).get("_value", term)
        summary_content = entry.get("summary", {}).get("_value", "")

        # Clean HTML from summary
        clean_summary = re.sub(r'<[^>]+>', '', summary_content)
        clean_summary = re.sub(r'\s+', ' ', clean_summary).strip()

        # Extract link
        links = entry.get("link", [])
        source_url = ""
        if links:
            source_url = links[0].get("href", "")

        return {
            "term": term,
            "definition": clean_summary[:2000] if clean_summary else f"{term} is a medical laboratory test.",
            "normal_range": "",  # MedlinePlus doesn't always provide this directly
            "clinical_significance": clean_summary[:500] if clean_summary else "",
            "source_url": source_url
        }

    except Exception as e:
        print(f"  [MedlinePlus] Error fetching '{term}': {e}")
        return None


def build_fetch(terms_list: list) -> list:
    """
    Fetch definitions for a list of medical terms from MedlinePlus.
    Called at KB build time (offline).

    Args:
        terms_list: list of medical term strings

    Returns:
        list of dicts with term definitions
    """
    os.makedirs(KB_RAW_DIR, exist_ok=True)

    results = []
    failed = []

    print(f"[MedlinePlus] Fetching {len(terms_list)} terms...")

    for i, term in enumerate(terms_list):
        result = _fetch_term(term)

        if result:
            results.append(result)
            # Save individual raw response
            safe_name = re.sub(r'[^\w\-]', '_', term.lower())
            filepath = os.path.join(KB_RAW_DIR, f"{safe_name}.json")
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
        else:
            failed.append(term)
            # Create a basic fallback entry so we have coverage
            results.append({
                "term": term,
                "definition": f"{term} is a medical laboratory test used in clinical diagnostics. Consult your healthcare provider for specific interpretation of your results.",
                "normal_range": "",
                "clinical_significance": f"{term} results help healthcare providers assess health status.",
                "source_url": ""
            })

        # Rate limiting — be gentle with the API
        if i % 10 == 9:
            print(f"  Fetched {i + 1}/{len(terms_list)} terms...")
            time.sleep(1)

    print(f"\n[MedlinePlus] Fetch complete:")
    print(f"  Successfully fetched: {len(results) - len(failed)}")
    print(f"  API misses (fallback used): {len(failed)}")

    if failed:
        print(f"  Failed terms: {', '.join(failed[:20])}{'...' if len(failed) > 20 else ''}")

    # Save combined results
    combined_path = os.path.join(KB_RAW_DIR, "_all_terms.json")
    with open(combined_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    return results


def live_fetch(term: str) -> dict:
    """
    Fetch a single term live from MedlinePlus (Tier 2 — called at query time).

    Args:
        term: medical term to look up

    Returns:
        dict with definition or None if not found
    """
    result = _fetch_term(term)
    if result and result.get("definition"):
        return result
    return None
