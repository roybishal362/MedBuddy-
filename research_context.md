# MedBuddy — Research Context Log
*Maintained during system build. Source for IEEE ESCI 2026 paper.*

## 1. Problem Statement
According to WHO (2023), approximately 80% of patients find medical lab reports difficult to understand due to technical jargon. The Journal of the American Medical Association (JAMA) reports that over 36% of U.S. adults have limited health literacy, leading to misinterpretation of critical lab values, delayed medical follow-ups, and increased healthcare costs.

Existing tools for medical text simplification either:
- Lack OCR capability for scanned reports
- Do not ground explanations in authoritative medical sources
- Have no hallucination detection mechanism
- Do not support conversational follow-up about reports
- Offer no adaptive literacy-level explanations

MedBuddy addresses all these gaps simultaneously.

## 2. Related Work
1. Lewis et al. (2020) — "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" — NeurIPS. DOI: 10.48550/arXiv.2005.11401
2. Ji et al. (2023) — "Survey of Hallucination in Natural Language Generation" — ACM Computing Surveys. DOI: 10.1145/3571730
3. Singhal et al. (2023) — "Large Language Models Encode Clinical Knowledge" — Nature. DOI: 10.1038/s41586-023-06291-2
4. Luo et al. (2022) — "BioGPT: Generative Pre-trained Transformer for Biomedical Text Generation" — Briefings in Bioinformatics. DOI: 10.1093/bib/bbac409
5. Guo et al. (2023) — "Evaluating Large Language Models on Medical Evidence Summarization" — NPJ Digital Medicine. DOI: 10.1038/s41746-023-00896-7
6. Srikanth & Li (2021) — "Elaborative Simplification" — ACL Findings. DOI: 10.18653/v1/2021.findings-acl.277
7. Touvron et al. (2023) — "LLaMA: Open and Efficient Foundation Language Models" — arXiv. DOI: 10.48550/arXiv.2302.13971

## 3. System Architecture Decisions
- **3-tier RAG over FAISS-only**: Ensures 100% query coverage — FAISS for cached terms, MedlinePlus API for live knowledge, LLM for long-tail terms with appropriate disclaimers
- **Lazy indexing in Tier 2**: Progressively enriches the knowledge base during runtime without manual re-indexing
- **Separate report_summarizer from entity_extractor**: Two-prompt design — Prompt A generates holistic narrative for user comprehension, Prompt B extracts structured data for programmatic processing. Running independently prevents one task from degrading the other
- **PyMuPDF + OCR fallback**: PyMuPDF handles digital PDFs efficiently; OCR with OpenCV preprocessing handles scanned reports. Hybrid approach maximizes coverage across real-world report formats
- **ConversationBufferWindowMemory k=5**: Limits context window to prevent prompt overflow while maintaining enough history for coherent multi-turn conversations
- **FALLBACK_MODEL for extraction, PRIMARY_MODEL for explanation**: Mixtral is faster and more reliable for structured JSON output; LLaMA 3.3 produces better narrative explanations
- **pdfplumber for digital tables, regex for OCR tables**: pdfplumber leverages PDF structure directly; OCR text lacks structure, so regex patterns are more suitable

## 4. Dataset & Knowledge Base
- Total KB terms indexed: 150+
- MedlinePlus API coverage rate: TBD (populated after build_kb.py run)
- Test set: 40 cases (10 Mode 1, 10 digital Mode 2, 10 OCR Mode 2, 5 OOV, 5 edge cases)
- Sample reports: 3 (digital PDF, scanned PDF, complex table PDF)

## 5. Experimental Results
*Populated after eval_pipeline.py runs*

## 6. Ablation Study Notes
*Populated after ablation runs*

## 7. Limitations
- OCR fails on very low-quality scans (below 200 DPI)
- Hindi OCR less accurate than English
- MedlinePlus covers US-standard tests; some regional tests may miss
- System does not handle multi-patient reports
- No image interpretation (X-ray, MRI reports not supported)
- Rate limits on Groq API may affect batch processing

## 8. Figure Descriptions
- Figure 1: Full system architecture diagram (2-mode pipeline)
- Figure 2: 3-tier RAG retrieval flowchart with decision logic
- Figure 3: Document intelligence pipeline (OCR flow)
- Figure 4: Evaluation results comparison bar chart (RAG vs baseline)
- Figure 5: Sample MedBuddy UI screenshot — report summary + table
- Figure 6: Grounding score distribution histogram across test set

## 9. Novel Contributions Summary
- C1: 3-tier adaptive RAG with lazy MedlinePlus indexing
- C2: Adaptive OCR pipeline for scanned + complex-table lab reports
- C3: Two-prompt design separating holistic comprehension from entity extraction
- C4: Conversational grounded Q&A with session-scoped hallucination control

## 10. Raw Metric Log
*Append every eval run here with timestamp — do not overwrite*
