"""
MedBuddy — Knowledge Base Builder
Fetches medical terms from MedlinePlus, embeds with all-MiniLM-L6-v2, stores in FAISS.
Run this script once before starting the application.
"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config.settings import EMBEDDING_MODEL, FAISS_INDEX_PATH, KB_TERMS
from knowledge_base.medlinePlus_fetcher import build_fetch


def build_knowledge_base():
    """Build the FAISS knowledge base from MedlinePlus data."""
    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import HuggingFaceEmbeddings

    print("=" * 60)
    print("MedBuddy — Knowledge Base Builder")
    print("=" * 60)

    # Step 1: Fetch terms from MedlinePlus
    print(f"\n[Step 1] Fetching {len(KB_TERMS)} medical terms from MedlinePlus...")
    results = build_fetch(KB_TERMS)
    print(f"  Total entries: {len(results)}")

    # Step 2: Prepare documents for embedding
    print("\n[Step 2] Preparing documents for embedding...")
    texts = []
    metadatas = []
    for entry in results:
        text_content = f"Term: {entry['term']}\n\nDefinition: {entry['definition']}"
        if entry.get('clinical_significance'):
            text_content += f"\n\nClinical Significance: {entry['clinical_significance']}"
        if entry.get('normal_range'):
            text_content += f"\n\nNormal Range: {entry['normal_range']}"

        texts.append(text_content)
        metadatas.append({
            "term": entry["term"],
            "source_url": entry.get("source_url", ""),
            "source": "MedlinePlus"
        })

    # Step 3: Create embeddings and FAISS index
    print(f"\n[Step 3] Creating embeddings with {EMBEDDING_MODEL}...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    print(f"\n[Step 4] Building FAISS index with {len(texts)} documents...")
    vector_store = FAISS.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas
    )

    # Step 5: Persist index
    os.makedirs(FAISS_INDEX_PATH, exist_ok=True)
    vector_store.save_local(FAISS_INDEX_PATH)

    print(f"\n[Step 5] FAISS index saved to: {FAISS_INDEX_PATH}")

    # Summary
    print("\n" + "=" * 60)
    print("Knowledge Base Build Complete!")
    print(f"  Total terms indexed: {len(texts)}")
    print(f"  Index location: {FAISS_INDEX_PATH}")
    print("=" * 60)

    # Verify with test queries
    print("\n[Verification] Testing retrieval...")
    test_terms = ["Hemoglobin", "Creatinine", "HbA1c", "LDL Cholesterol", "TSH"]
    for term in test_terms:
        docs = vector_store.similarity_search_with_score(term, k=1)
        if docs:
            doc, score = docs[0]
            similarity = 1 / (1 + score)
            matched_term = doc.metadata.get("term", "unknown")
            print(f"  Query: '{term}' → Match: '{matched_term}' (similarity: {similarity:.3f})")

    return vector_store


if __name__ == "__main__":
    build_knowledge_base()
