"""
MedBuddy — Wikipedia Live Fetcher
Real-time, broad-coverage knowledge tier. Unlike the blood-test FAISS index,
this covers virtually any medical/anatomical/imaging/pathology term, with a
citable source URL. Wrapped defensively so a network failure degrades to the
LLM tier instead of breaking the app.
"""
import re
import requests

WIKI_API = "https://en.wikipedia.org/w/api.php"
HEADERS = {"User-Agent": "MedBuddy/1.0 (educational medical report explainer)"}
DEFAULT_TIMEOUT = 6


def _search_title(term: str, timeout: int = DEFAULT_TIMEOUT):
    """Resolve a free-text term to the best-matching Wikipedia article title."""
    params = {
        "action": "opensearch",
        "search": term,
        "limit": 1,
        "namespace": 0,
        "format": "json",
    }
    resp = requests.get(WIKI_API, params=params, headers=HEADERS, timeout=timeout)
    if resp.status_code != 200:
        return None
    data = resp.json()
    titles = data[1] if isinstance(data, list) and len(data) > 1 else []
    return titles[0] if titles else None


def _get_extract(title: str, timeout: int = DEFAULT_TIMEOUT):
    """Fetch the plain-text intro extract for a Wikipedia article title."""
    params = {
        "action": "query",
        "prop": "extracts",
        "exintro": 1,
        "explaintext": 1,
        "redirects": 1,
        "format": "json",
        "titles": title,
    }
    resp = requests.get(WIKI_API, params=params, headers=HEADERS, timeout=timeout)
    if resp.status_code != 200:
        return None, None
    pages = resp.json().get("query", {}).get("pages", {})
    for _, page in pages.items():
        extract = page.get("extract", "")
        if extract and extract.strip():
            pageid = page.get("pageid")
            if pageid:
                url = f"https://en.wikipedia.org/?curid={pageid}"
            else:
                url = "https://en.wikipedia.org/wiki/" + title.replace(" ", "_")
            return extract, url
    return None, None


def wikipedia_fetch(term: str, timeout: int = DEFAULT_TIMEOUT) -> dict:
    """
    Look up a term on Wikipedia in real time.

    Returns:
      {"term", "definition", "source_url", "source": "Wikipedia"} or None.
    """
    if not term or not str(term).strip():
        return None
    try:
        title = _search_title(term, timeout=timeout)
        if not title:
            return None
        extract, url = _get_extract(title, timeout=timeout)
        if not extract:
            return None
        definition = re.sub(r"\s+", " ", extract).strip()[:2000]
        if not definition:
            return None
        return {
            "term": term,
            "definition": definition,
            "source_url": url,
            "source": "Wikipedia",
        }
    except Exception as e:
        print(f"[Wikipedia] fetch error for '{term}': {e}")
        return None
