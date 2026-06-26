# MedBuddy 🩺

**Conversational AI for Medical Report Simplification**

MedBuddy helps patients understand their medical lab reports in plain language using AI-powered simplification, OCR for scanned reports, and a 3-tier RAG knowledge retrieval system.

## Features

- **Mode 1 — Ask a Term**: Query any medical term and get a plain-language explanation
- **Mode 2 — Upload Report**: Upload a PDF lab report for comprehensive analysis
  - Automatic text extraction (digital + scanned PDF via OCR)
  - Narrative summary at your chosen literacy level
  - Per-entity explanations with risk flagging (Normal/Borderline/Critical)
  - Conversational Q&A about your report
- **3-Tier RAG**: FAISS cached knowledge → MedlinePlus live API → LLM fallback
- **Hallucination Verification**: Every explanation is grounded and scored
- **Bilingual**: English + Hindi support
- **Adaptive Literacy Levels**: Basic, Intermediate, Advanced

## Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM | Groq API (LLaMA 3.3 + Mixtral) |
| Orchestration | LangChain |
| Embeddings | all-MiniLM-L6-v2 (sentence-transformers) |
| Vector Store | FAISS |
| PDF Parsing | PyMuPDF + pdfplumber |
| OCR | pytesseract + OpenCV |
| Medical KB | MedlinePlus Connect API |
| UI | Streamlit |

## Quick Start

### 1. System Dependencies

```bash
# Windows (with chocolatey)
choco install tesseract poppler

# Linux / WSL
sudo apt install tesseract-ocr tesseract-ocr-hin poppler-utils

# macOS
brew install tesseract poppler
```

### 2. Python Setup

```bash
cd medbuddy
pip install -r requirements.txt
```

### 3. Environment Variables

```bash
# Create .env file from template
cp .env.example .env
# Edit .env and add your Groq API key
# Get your key at: https://console.groq.com/keys
```

### 4. Build Knowledge Base (One-Time)

```bash
python knowledge_base/build_kb.py
```

This fetches 150+ medical term definitions from MedlinePlus and creates the FAISS index.
Takes approximately 5-10 minutes on first run.

### 5. Run the Application

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

## Deploy to Streamlit Community Cloud

The repo is pre-configured for [share.streamlit.io](https://share.streamlit.io):

- `packages.txt` installs the system OCR binaries (Tesseract + Poppler).
- `requirements.txt` pins a CPU-only PyTorch so the build stays within Cloud limits.
- `.streamlit/config.toml` carries the theme and upload-size settings.
- The prebuilt FAISS index in `knowledge_base/faiss_index/` is committed, so **no KB build step is needed** in the cloud.

**Steps:**

1. Push this folder to a GitHub repo (this `README.md`, `app.py`, etc. at the repo root).
2. On [share.streamlit.io](https://share.streamlit.io), click **Create app** → pick your repo/branch.
3. Set **Main file path** to `app.py` and **Python version** to `3.12`.
4. Open **Advanced settings → Secrets** and add:
   ```toml
   GROQ_API_KEY = "your_real_groq_key"
   ```
5. Click **Deploy**. First build takes a few minutes (Torch + the embedding model download on first run).

> The free tier provides ~1 GB RAM. Torch + sentence-transformers + FAISS are close to that ceiling; if the app reboots under load, that's the usual cause.

## Running Evaluation

```bash
python evaluation/eval_pipeline.py
```

Results are saved to `evaluation/results/`.

## Project Structure

```
medbuddy/
├── app.py                          # Streamlit main entry point
├── requirements.txt
├── .env.example
├── research_context.md
├── README.md
├── config/
│   └── settings.py                 # All config, prompts, constants
├── core/
│   ├── intent_classifier.py        # Mode 1: query intent detection
│   ├── document_intelligence.py    # Mode 2: PDF + OCR extraction
│   ├── report_analyzer.py          # Mode 2: LLM report comprehension
│   ├── report_summarizer.py        # Mode 2: narrative summary
│   ├── entity_extractor.py         # Mode 2: structured entity extraction
│   ├── value_contextualizer.py     # Mode 2: NORMAL/BORDERLINE/CRITICAL flags
│   ├── rag_retriever.py            # 3-tier FAISS + live API + LLM fallback
│   ├── adaptive_explainer.py       # Literacy-level explanation generation
│   ├── hallucination_verifier.py   # Grounding verification
│   ├── conversation_memory.py      # Session memory for report Q&A
│   └── refusal_handler.py          # Safety: out-of-scope blocking
├── knowledge_base/
│   ├── build_kb.py
│   ├── medlinePlus_fetcher.py
│   ├── raw/
│   └── faiss_index/
├── evaluation/
│   ├── eval_pipeline.py
│   ├── metrics.py
│   ├── test_cases.json
│   └── results/
├── paper/
│   └── medbuddy_ieee.tex
└── assets/
    └── sample_reports/
```

## License

For academic and research use. IEEE ESCI 2026 publication.
