"""
MedBuddy — RAG Retriever
3-tier fallback: FAISS → MedlinePlus Live → LLM Fallback
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config.settings import SIMILARITY_THRESHOLD, EMBEDDING_MODEL, FAISS_INDEX_PATH


# Module-level cache for the FAISS index
_faiss_store = None
_embeddings = None


def _get_embeddings():
    """Lazy-load embeddings model."""
    global _embeddings
    if _embeddings is None:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        _embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return _embeddings


def _get_faiss_store():
    """Lazy-load FAISS index from disk."""
    global _faiss_store
    if _faiss_store is None:
        from langchain_community.vectorstores import FAISS
        if os.path.exists(FAISS_INDEX_PATH):
            _faiss_store = FAISS.load_local(
                FAISS_INDEX_PATH,
                _get_embeddings(),
                allow_dangerous_deserialization=True
            )
        else:
            print(f"[RAGRetriever] WARNING: FAISS index not found at {FAISS_INDEX_PATH}")
            print("  Run 'python knowledge_base/build_kb.py' first!")
    return _faiss_store


def retrieve(term: str) -> dict:
    """
    Retrieve relevant medical knowledge using 3-tier fallback.

    Returns:
      {
        "context": str | None,
        "source": "FAISS" | "MedlinePlus_live" | "LLM_fallback",
        "score": float | None,
        "source_url": str | None,
        "disclaimer": bool
      }
    """
    faiss_store = _get_faiss_store()

    # ─── TIER 1: Static FAISS search ──────────────────────────────
    if faiss_store is not None:
        try:
            results = faiss_store.similarity_search_with_score(term, k=1)
            if results:
                top_doc, score = results[0]
                # FAISS returns L2 distance — lower is better
                # Normalize: similarity = 1 / (1 + score)
                similarity = 1 / (1 + score)

                if similarity >= SIMILARITY_THRESHOLD:
                    return {
                        "context": top_doc.page_content,
                        "source": "FAISS",
                        "score": round(similarity, 4),
                        "source_url": top_doc.metadata.get("source_url"),
                        "disclaimer": False
                    }
        except Exception as e:
            print(f"[RAGRetriever] FAISS search error: {e}")

    # ─── TIER 2: Live MedlinePlus fetch ───────────────────────────
    try:
        from knowledge_base.medlinePlus_fetcher import live_fetch
        live_result = live_fetch(term)

        if live_result and live_result.get("definition"):
            # Lazy-add to FAISS for future queries
            if faiss_store is not None:
                try:
                    text_content = f"Term: {live_result['term']}\n\nDefinition: {live_result['definition']}"
                    faiss_store.add_texts(
                        [text_content],
                        metadatas=[{
                            "term": term,
                            "source_url": live_result.get("source_url", ""),
                            "source": "MedlinePlus_live"
                        }]
                    )
                    faiss_store.save_local(FAISS_INDEX_PATH)
                except Exception as e:
                    print(f"[RAGRetriever] Failed to lazy-add to FAISS: {e}")

            return {
                "context": live_result["definition"],
                "source": "MedlinePlus_live",
                "score": None,
                "source_url": live_result.get("source_url"),
                "disclaimer": False
            }
    except Exception as e:
        print(f"[RAGRetriever] Tier 2 (MedlinePlus live) error: {e}")

    # ─── TIER 3: LLM general knowledge fallback ──────────────────
    return {
        "context": None,
        "source": "LLM_fallback",
        "score": None,
        "source_url": None,
        "disclaimer": True
    }


def reload_index():
    """Force reload the FAISS index from disk."""
    global _faiss_store
    _faiss_store = None
    return _get_faiss_store()
