"""
MedBuddy Configuration
All constants, model names, prompt templates, and system settings.
"""
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()

# ─── Model Configuration ───────────────────────────────────────────
PRIMARY_MODEL = os.getenv("PRIMARY_MODEL", "llama-3.3-70b-versatile").strip()
FALLBACK_MODEL = os.getenv("FALLBACK_MODEL", "llama3-70b-8192").strip()

# LLM Instances
primary_llm = ChatGroq(model=PRIMARY_MODEL, temperature=0, api_key=GROQ_API_KEY)
fallback_llm = ChatGroq(model=FALLBACK_MODEL, temperature=0, api_key=GROQ_API_KEY)

# ─── RAG Configuration ─────────────────────────────────────────────
SIMILARITY_THRESHOLD = 0.40
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
FAISS_INDEX_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "knowledge_base", "faiss_index")
KB_RAW_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "knowledge_base", "raw")

# ─── OCR Configuration ─────────────────────────────────────────────
OCR_TEXT_MIN_LENGTH = 100
OCR_DPI = 300

# ─── Language & Literacy ───────────────────────────────────────────
SUPPORTED_LANGUAGES = ["English", "Hindi"]

LITERACY_LEVELS = {
    "Basic": "Explain as if to a 5th grade student. No jargon. Very short sentences. Max 80 words.",
    "Intermediate": "Explain to a high school student. Define technical terms immediately after use. Max 120 words.",
    "Advanced": "Explain to a college-educated adult. Technical language acceptable. Max 150 words."
}

LITERACY_WORD_LIMITS = {
    "Basic": 80,
    "Intermediate": 120,
    "Advanced": 150
}

# ─── Refusal Trigger Keywords ──────────────────────────────────────
# Personal-advice intent only. Defining a treatment/medication/condition is
# allowed (educational); recommending one FOR THE USER is refused.
REFUSAL_KEYWORDS = [
    "diagnose me", "diagnose my", "what disease do i", "what condition do i",
    "prescribe", "prescription for", "what should i do", "what should i take",
    "should i take", "should i stop", "do i have", "is this cancer",
    "am i sick", "recommend treatment", "recommend a medicine", "cure for my",
    "how do i treat", "how to treat my", "which medicine should",
]

# ─── Refusal Response Templates ────────────────────────────────────
REFUSAL_RESPONSE = {
    "English": (
        "MedBuddy is designed to help you understand medical terms and reports in "
        "plain language. For personal medical advice, diagnosis, or treatment decisions, "
        "please consult a qualified healthcare professional. I'm happy to explain what any "
        "medical word, test, condition, or result means."
    ),
    "Hindi": (
        "MedBuddy चिकित्सा शब्दों और रिपोर्ट को सरल भाषा में समझने में मदद करता है। "
        "व्यक्तिगत चिकित्सा सलाह, निदान, या उपचार के लिए कृपया किसी योग्य स्वास्थ्य पेशेवर से परामर्श करें। "
        "मैं किसी भी चिकित्सा शब्द, जांच, स्थिति या परिणाम का अर्थ समझाने में खुश हूँ।"
    )
}

# ─── Prompt Templates ──────────────────────────────────────────────

# Intent Classifier Prompt
INTENT_CLASSIFIER_PROMPT = """You are a medical query intent classifier.

Classify the following user query into exactly one category:
1. MEDICAL_TERM — the query asks what a medical or health term means: a lab test, a disease/condition, body part/anatomy, a symptom, a medical procedure, an imaging scan, a medication or treatment as a concept, or any general health term (e.g., Hemoglobin, Hypertension, MRI, Biopsy, Tachycardia, Edema, Chemotherapy, Cholelithiasis)
2. REPORT_QUESTION — the query asks about the patient's own medical report when none has been uploaded (e.g., "what does my report say?")
3. OUT_OF_SCOPE — the query asks for PERSONAL medical advice (a diagnosis for the user, a treatment/medication recommendation for the user, "what should I do", "do I have X"), or is not about medicine or health at all

User query: {query}

Return ONLY valid JSON. No explanation. No markdown. No preamble.
{{
  "intent": "<MEDICAL_TERM|REPORT_QUESTION|OUT_OF_SCOPE>",
  "confidence": <float 0.0 to 1.0>,
  "normalized_term": "<standardized medical term if MEDICAL_TERM, else null>"
}}"""

# Report Analyzer Prompt
REPORT_ANALYZER_PROMPT = """You are a medical report analysis engine.

Analyze the following lab report text and produce a structured internal analysis.

Report:
{structured_report_text}

Return ONLY valid JSON. No explanation. No markdown.
{{
  "overall_assessment": "<brief overall assessment>",
  "critical_findings": [<list of critical or abnormal findings>],
  "normal_findings": [<list of normal findings>],
  "report_type": "<type of report, e.g., Complete Blood Count>",
  "patient_indicators": {{}}
}}"""

# Report Summarizer Prompt (Prompt A)
REPORT_SUMMARIZER_PROMPT = """System: You are MedBuddy, a compassionate medical report interpreter helping patients understand their lab results in plain language.

Below is a patient's lab report. Analyze it fully and write a clear, structured summary.

Report:
{structured_report_text}

Internal Analysis:
{report_analyzer_output}

Write your summary for a patient with {literacy_level} health literacy.

Your summary MUST follow this structure:
1. Overall Assessment (1 sentence — honest but not alarming)
2. What looks normal (brief, reassuring)
3. What needs attention — for EACH abnormal value:
   - What the test measures (1 sentence)
   - What the patient's specific result means (1 sentence)
   - Why it matters (1 sentence, no diagnosis)
4. Closing recommendation (always advise consulting their doctor)

Rules:
- Do NOT diagnose any condition
- Do NOT recommend any medication or treatment
- Do NOT alarm the patient unnecessarily
- Do NOT add any information not present in the report
- Language: {language}
- Literacy level: {literacy_level_description}"""

# Entity Extractor Prompt (Prompt B)
ENTITY_EXTRACTOR_PROMPT = """You are a medical data extraction engine.

Extract ALL lab test results from the report text below.
Return ONLY a valid JSON array. No explanation. No markdown. No preamble.

Report text:
{structured_report_text}

For each test result extract:
- term: full test name
- abbreviation: short form if present, else null
- patient_value: the patient's result value as a string
- unit: measurement unit
- reference_range: as shown in report (e.g. "13.5-17.5")
- reference_low: lower bound as float
- reference_high: upper bound as float

If any field is not present in the report, set it to null.
Extract ONLY what is present. Do not infer or add values.
Return JSON array only."""

# RAG-Grounded Explanation Prompt (Tier 1 + 2)
RAG_EXPLANATION_PROMPT = """System: You are a medical communication specialist. Your task is to explain a medical lab result to a patient using ONLY the information provided in the context below. Do not add any medical claims not supported by the context.

Context (from MedlinePlus):
{retrieved_context}

Patient result: {term} = {value} {unit}
Reference range: {ref_range}
Status: {flag}

Literacy Level instruction: {literacy_level_description}
Language: {language}

Write a clear, compassionate explanation covering:
1. What this test measures (1-2 sentences)
2. What the patient's specific result means (1-2 sentences)
3. A simple analogy only if it genuinely aids understanding (optional)

Important:
- If the context above does not actually describe "{term}" (for example it is about a different test), do NOT use it. Instead explain "{term}" accurately from your own general medical knowledge.
- Be specific and factual. Avoid vague filler such as "it helps doctors know how you're doing".

Do NOT: diagnose, recommend medication, suggest treatment.
Do NOT: use information not in the context above.
Respond in {language} only."""

# LLM Fallback Explanation Prompt (Tier 3)
LLM_FALLBACK_PROMPT = """System: You are a medical communication specialist.
This term was not found in the verified knowledge base.
Provide a careful general explanation based on your knowledge.

Term: {term}
Patient result: {value} {unit} (Reference: {ref_range}) — Status: {flag}
Literacy Level: {literacy_level_description}
Language: {language}

Write a brief explanation. Begin your response with:
"[Note: This explanation is based on general medical knowledge and has not been verified against a certified source. Please confirm with your doctor.]"

Do NOT diagnose or recommend treatment."""

# Dictionary-Style Term Explanation (Mode 1, no patient value) — RAG-grounded
TERM_DICTIONARY_PROMPT = """System: You are MedBuddy, explaining a medical term to a patient who simply asked what it means — like a friendly medical dictionary. The term may be a lab test, a condition or disease, a body part, a symptom, a procedure, an imaging scan, or any health concept. The patient has NOT given a personal result, so do NOT refer to "your result", do NOT invent a value, and do NOT ask them for one.

Context (reference):
{retrieved_context}

Medical term: {term}

Literacy Level instruction: {literacy_level_description}
Language: {language}

Write a clear, compassionate explanation covering:
1. What "{term}" means in simple words — what it is or what it refers to.
2. Why it matters or what it is used for (1-2 sentences).
3. ONLY if "{term}" is a measurable lab test, also give the typical normal / reference range for a healthy adult, and note that ranges vary by lab, age, and sex. If it is a condition, body part, symptom, or procedure, skip the range.
4. A simple analogy ONLY if it genuinely aids understanding (optional).

Do NOT: diagnose the patient, recommend medication or treatment for them, or invent a personal result.
Prefer information supported by the context above; if the context does not fit "{term}", use your own general medical knowledge.
Respond in {language} only."""

# Dictionary-Style Term Explanation (Mode 1, no patient value) — Tier 4 fallback
TERM_DICTIONARY_FALLBACK_PROMPT = """System: You are MedBuddy, a medical communication specialist explaining a medical term like a friendly dictionary. The term may be a lab test, condition, body part, symptom, procedure, or any health concept. The patient has NOT given a personal result, so do NOT refer to a personal value or ask for one.

Medical term: {term}
Literacy Level: {literacy_level_description}
Language: {language}

Cover:
1. What "{term}" means in simple words.
2. Why it matters or what it is used for.
3. ONLY if "{term}" is a measurable lab test, give the typical normal range (note that ranges vary by lab, age, and sex); otherwise skip the range.

Do NOT diagnose or recommend treatment. Respond in {language} only."""

# Hallucination Verification Prompt
HALLUCINATION_VERIFICATION_PROMPT = """You are a medical fact-checker. You receive a retrieved medical context and a generated patient explanation.

Retrieved Context:
{context}

Generated Explanation:
{explanation}

Task: Identify every factual medical claim in the explanation.
For each claim, determine: is it supported by the retrieved context?

Return ONLY valid JSON. No preamble. No markdown.
{{
  "grounding_score": <float 0.0 to 1.0, proportion of claims supported>,
  "ungrounded_claims": [<list of specific unsupported claim strings>],
  "verdict": "PASS" if grounding_score >= 0.85 else "FLAG"
}}"""

# Conversation Follow-up Prompt (Prompt C)
CONVERSATION_FOLLOWUP_PROMPT = """You are MedBuddy, a medical report assistant.

The patient has uploaded a report. Here is the complete context:

Report Summary:
{narrative_summary}

Detailed Entities:
{entities_with_flags_and_explanations}

How to answer:
- If the question is about the patient's own results or findings, answer using the report context above.
- If the question is a general "what is ___" about a medical term, test, body part, or concept (e.g. "what is a uterus", "what is gravida", "what does consanguinity mean"), give a brief, clear, plain-language definition from general medical knowledge — even if it is not in the report. Keep it general and educational; do NOT tie it to the patient's specific values.
- Only reply "That information is not available in your report. Please ask your doctor." when the patient asks for a specific result, value, or finding that genuinely is not present in the report above.

Safety rules (always apply):
- Never diagnose, never prescribe, never recommend treatment, and never predict outcomes.
- If asked for a diagnosis or treatment, gently redirect them to their doctor.
- Be warm, clear, and helpful.
- Language: {language}

Chat History:
{chat_history}

Patient's question: {user_question}

Answer now in {language}."""

# ─── Report Kind Classifier (LAB vs NARRATIVE) ─────────────────────
REPORT_KIND_PROMPT = """You are a medical report classifier.

Report (excerpt):
{report_text}

Classify the report into exactly one category. Return ONLY valid JSON. No markdown. No preamble.
{{"kind": "<LAB|NARRATIVE>"}}

Definitions:
- LAB = a laboratory test report built around numeric results with reference ranges (e.g. CBC, lipid profile, liver/kidney panel, blood sugar, thyroid, urine values).
- NARRATIVE = any report described mainly in words rather than numeric ranges: radiology / imaging (ultrasound, sonography, X-ray, CT, MRI), pathology or biopsy, discharge summary, prescription, or clinical notes."""

# ─── Narrative Report Explainer (imaging / pathology / notes) ──────
NARRATIVE_REPORT_PROMPT = """System: You are MedBuddy, helping a patient understand a medical report in plain language. This is a NARRATIVE report (for example an ultrasound / sonography, X-ray, CT, MRI, pathology / biopsy, discharge summary, or clinical note) — NOT a numeric lab panel.

Report:
{report_text}

Literacy level instruction: {literacy_level_description}
Respond in {language}.

Return ONLY valid JSON. No markdown. No preamble.
{{
  "summary": "<3-5 sentence plain-language overview of what this report is and what it says — honest but not alarming>",
  "findings": [
    {{"finding": "<short label of a key finding stated in the report>", "meaning": "<1-2 sentence plain-language meaning, no diagnosis>"}}
  ],
  "glossary": [
    {{"term": "<a medical or technical word that appears in the report>", "definition": "<one simple sentence defining it>"}}
  ]
}}

Rules:
- Use ONLY information present in the report for "summary" and "findings". Never invent findings.
- "glossary" may use general medical knowledge to define terms that appear in the report.
- Do NOT diagnose, do NOT recommend medication or treatment, do NOT predict outcomes.
- Keep every explanation in plain {language}."""

# ─── UI Text Labels ────────────────────────────────────────────────
UI_LABELS = {
    "English": {
        "app_title": "🩺 MedBuddy",
        "app_subtitle": "Understand your health in plain language",
        "literacy_label": "Literacy Level",
        "language_label": "Language",
        "tab1_title": "🔍 Ask a Term",
        "tab2_title": "📄 Upload Report",
        "term_input_placeholder": "Enter any medical term (e.g. Hemoglobin, Hypertension, MRI, Biopsy)...",
        "explain_button": "Explain",
        "upload_label": "Upload your lab report (PDF or image)",
        "analyze_button": "🔬 Analyze Report",
        "analyzing_msg": "Reading your report...",
        "summary_header": "📊 Your Report Summary",
        "results_header": "📋 Detailed Results",
        "chat_header": "💬 Ask about your report",
        "chat_placeholder": "Ask me anything about your report...",
        "download_button": "📥 Download Summary as PDF",
        "about_title": "ℹ️ About MedBuddy",
        "about_text": (
            "MedBuddy helps you understand any medical term or report — lab tests, "
            "conditions, scans, procedures — in plain language. It does NOT provide "
            "medical advice, diagnosis, or treatment recommendations."
        ),
        "col_test": "Test",
        "col_value": "Your Value",
        "col_unit": "Unit",
        "col_status": "Status",
        "col_explanation": "What it means",
        "source_label": "Source",
        "grounded_label": "Grounded",
        "verified_label": "✅ Verified",
        "unverified_label": "⚠️ Unverified",
        "digital_badge": "📄 Digital PDF",
        "ocr_badge": "🔍 Scanned PDF (OCR)",
        "new_report_msg": "New report uploaded. Previous chat cleared.",
        "normal": "🟢 NORMAL",
        "borderline": "🟡 BORDERLINE",
        "critical": "🔴 CRITICAL",
        "unknown": "⚪ UNKNOWN",
    },
    "Hindi": {
        "app_title": "🩺 MedBuddy",
        "app_subtitle": "अपनी सेहत को सरल भाषा में समझें",
        "literacy_label": "साक्षरता स्तर",
        "language_label": "भाषा",
        "tab1_title": "🔍 शब्द पूछें",
        "tab2_title": "📄 रिपोर्ट अपलोड करें",
        "term_input_placeholder": "कोई भी चिकित्सा शब्द दर्ज करें (जैसे हीमोग्लोबिन, हाइपरटेंशन, MRI, बायोप्सी)...",
        "explain_button": "समझाएं",
        "upload_label": "अपनी लैब रिपोर्ट अपलोड करें (PDF या इमेज)",
        "analyze_button": "🔬 रिपोर्ट का विश्लेषण करें",
        "analyzing_msg": "आपकी रिपोर्ट पढ़ रहे हैं...",
        "summary_header": "📊 आपकी रिपोर्ट का सारांश",
        "results_header": "📋 विस्तृत परिणाम",
        "chat_header": "💬 अपनी रिपोर्ट के बारे में पूछें",
        "chat_placeholder": "अपनी रिपोर्ट के बारे में कुछ भी पूछें...",
        "download_button": "📥 सारांश PDF डाउनलोड करें",
        "about_title": "ℹ️ MedBuddy के बारे में",
        "about_text": (
            "MedBuddy किसी भी चिकित्सा शब्द या रिपोर्ट — जांच, स्थिति, स्कैन, प्रक्रिया — को "
            "सरल भाषा में समझने में मदद करता है। यह चिकित्सा सलाह, निदान या उपचार की सिफारिश नहीं करता।"
        ),
        "col_test": "जांच",
        "col_value": "आपका मान",
        "col_unit": "इकाई",
        "col_status": "स्थिति",
        "col_explanation": "इसका क्या मतलब है",
        "source_label": "स्रोत",
        "grounded_label": "आधारित",
        "verified_label": "✅ सत्यापित",
        "unverified_label": "⚠️ असत्यापित",
        "digital_badge": "📄 डिजिटल PDF",
        "ocr_badge": "🔍 स्कैन की गई PDF (OCR)",
        "new_report_msg": "नई रिपोर्ट अपलोड की गई। पिछली चैट हटा दी गई।",
        "normal": "🟢 सामान्य",
        "borderline": "🟡 सीमारेखा",
        "critical": "🔴 गंभीर",
        "unknown": "⚪ अज्ञात",
    }
}

# ─── Lab Test Terms for Knowledge Base ─────────────────────────────
KB_TERMS = [
    "Hemoglobin", "Hematocrit", "Red Blood Cell Count", "White Blood Cell Count",
    "Platelet Count", "Mean Corpuscular Volume", "Mean Corpuscular Hemoglobin",
    "Mean Corpuscular Hemoglobin Concentration", "Red Cell Distribution Width",
    "Neutrophils", "Lymphocytes", "Monocytes", "Eosinophils", "Basophils",
    "Fasting Blood Sugar", "Random Blood Sugar", "HbA1c", "Glycated Hemoglobin",
    "Oral Glucose Tolerance Test", "Insulin",
    "Blood Urea Nitrogen", "Creatinine", "Uric Acid", "eGFR",
    "Glomerular Filtration Rate", "Sodium", "Potassium", "Chloride",
    "Calcium", "Phosphorus", "Magnesium", "Bicarbonate",
    "Total Cholesterol", "LDL Cholesterol", "HDL Cholesterol", "Triglycerides",
    "VLDL Cholesterol", "Total to HDL Ratio",
    "AST", "ALT", "Alkaline Phosphatase", "GGT", "Total Bilirubin",
    "Direct Bilirubin", "Indirect Bilirubin", "Albumin", "Total Protein",
    "Globulin", "A/G Ratio",
    "TSH", "Free T3", "Free T4", "Total T3", "Total T4", "Anti-TPO Antibodies",
    "Vitamin D", "Vitamin B12", "Folate", "Iron", "Ferritin", "TIBC",
    "Transferrin Saturation",
    "PSA", "CA-125", "CEA", "AFP",
    "Prothrombin Time", "INR", "aPTT", "D-Dimer", "Fibrinogen",
    "ESR", "CRP", "C-Reactive Protein", "ANA", "Rheumatoid Factor",
    "Troponin", "CK-MB", "BNP", "NT-proBNP", "LDH",
    "Amylase", "Lipase",
    "Testosterone", "Estradiol", "Progesterone", "LH", "FSH", "Prolactin",
    "Cortisol", "DHEA-S",
    "HIV Antibody", "Hepatitis B Surface Antigen", "Hepatitis C Antibody",
    "RPR", "VDRL",
    "Urine Routine", "Urine Culture", "Urine Protein", "Urine Creatinine",
    "Microalbumin", "Albumin Creatinine Ratio",
    "Blood Group", "Rh Factor", "Direct Coombs Test", "Indirect Coombs Test",
    "Reticulocyte Count", "Peripheral Blood Smear",
    "Glucose Tolerance Test", "Fructosamine",
    "Homocysteine", "Lipoprotein(a)", "Apolipoprotein A1", "Apolipoprotein B",
    "Ceruloplasmin", "Copper",
    "Zinc", "Selenium", "Manganese",
    "Ammonia", "Lactate",
    "Parathyroid Hormone", "Calcitonin",
    "IGF-1", "Growth Hormone",
    "Anti-CCP", "Complement C3", "Complement C4",
    "IgA", "IgG", "IgM", "IgE",
    "Protein Electrophoresis", "Serum Free Light Chains",
    "Beta-2 Microglobulin",
    "Procalcitonin", "IL-6",
    "Troponin I", "Troponin T",
    "HBsAg", "Anti-HCV", "Anti-HAV",
    "Malaria Antigen", "Dengue NS1", "Dengue IgM",
    "Widal Test", "Blood Culture",
    "Semen Analysis", "Sperm Count",
    "AMH", "Anti-Mullerian Hormone",
    "Thyroglobulin", "Anti-Thyroglobulin Antibodies",
    "Pancreatic Elastase",
    "Stool Occult Blood", "Stool Routine",
    "ABG", "Arterial Blood Gas",
    "Osmolality", "Anion Gap",
    "Direct LDL", "Small Dense LDL",
    "Cystatin C", "Beta HCG",
]
