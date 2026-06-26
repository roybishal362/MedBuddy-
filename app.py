"""
MedBuddy — Streamlit Main Application
Two-mode medical report simplification system with conversational Q&A.
"""
import streamlit as st
import os
import sys
import json
import tempfile

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from config.settings import (
    LITERACY_LEVELS, SUPPORTED_LANGUAGES, UI_LABELS, GROQ_API_KEY
)
from core.intent_classifier import classify
from core.refusal_handler import handle_refusal, handle_report_redirect
from core.rag_retriever import retrieve
from core.adaptive_explainer import explain
from core.hallucination_verifier import verify
from core.document_intelligence import extract
from core.report_analyzer import analyze
from core.report_summarizer import summarize
from core.entity_extractor import extract_entities
from core.value_contextualizer import add_flags_to_entities
from core.conversation_memory import ConversationMemory

# ─── Page Configuration ────────────────────────────────────────────
st.set_page_config(
    page_title="MedBuddy — Understand Your Health",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main theme */
    .stApp {
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a3e 50%, #0d1b2a 100%);
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a3e 0%, #0f0f23 100%);
        border-right: 1px solid rgba(100, 100, 255, 0.15);
    }

    /* Cards */
    .summary-card {
        background: linear-gradient(135deg, rgba(30, 30, 80, 0.8), rgba(20, 20, 60, 0.9));
        border: 1px solid rgba(100, 100, 255, 0.2);
        border-radius: 16px;
        padding: 24px;
        margin: 16px 0;
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }

    .result-card {
        background: rgba(25, 25, 65, 0.7);
        border: 1px solid rgba(100, 100, 255, 0.15);
        border-radius: 12px;
        padding: 16px;
        margin: 8px 0;
    }

    .explanation-card {
        background: rgba(20, 20, 55, 0.6);
        border-left: 4px solid #6366f1;
        border-radius: 0 12px 12px 0;
        padding: 16px;
        margin: 8px 0;
    }

    /* Badges */
    .badge-verified {
        background: linear-gradient(135deg, #059669, #10b981);
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8em;
        font-weight: 600;
    }
    .badge-unverified {
        background: linear-gradient(135deg, #d97706, #f59e0b);
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8em;
        font-weight: 600;
    }
    .badge-digital {
        background: linear-gradient(135deg, #2563eb, #3b82f6);
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8em;
    }
    .badge-ocr {
        background: linear-gradient(135deg, #7c3aed, #8b5cf6);
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8em;
    }

    /* Flag colors */
    .flag-normal { color: #10b981; font-weight: 700; }
    .flag-borderline { color: #f59e0b; font-weight: 700; }
    .flag-critical { color: #ef4444; font-weight: 700; }
    .flag-unknown { color: #6b7280; font-weight: 700; }

    /* Chat messages */
    .chat-user {
        background: rgba(99, 102, 241, 0.2);
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 16px 16px 4px 16px;
        padding: 12px 16px;
        margin: 8px 0;
        max-width: 80%;
        margin-left: auto;
    }
    .chat-bot {
        background: rgba(30, 30, 80, 0.6);
        border: 1px solid rgba(100, 100, 255, 0.15);
        border-radius: 16px 16px 16px 4px;
        padding: 12px 16px;
        margin: 8px 0;
        max-width: 80%;
    }

    /* Disclaimer */
    .disclaimer-box {
        background: rgba(217, 119, 6, 0.15);
        border: 1px solid rgba(245, 158, 11, 0.4);
        border-radius: 12px;
        padding: 12px 16px;
        margin: 8px 0;
        font-size: 0.9em;
    }

    /* Title styling */
    .main-title {
        background: linear-gradient(135deg, #818cf8, #6366f1, #4f46e5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.2em;
        font-weight: 800;
    }

    /* Hide default streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    div[data-testid="stStatusWidget"] {display: none;}
</style>
""", unsafe_allow_html=True)

# ─── Presentation polish (additive — no logic touched) ─────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"], .stApp, button, input, textarea {
        font-family: 'Inter', -apple-system, 'Segoe UI', sans-serif;
    }

    .block-container { padding-top: 2.2rem; max-width: 1180px; }

    /* ── Hero header ──────────────────────────────────────────── */
    .mb-hero {
        background:
            radial-gradient(1200px 200px at 15% -40%, rgba(99,102,241,0.35), transparent 60%),
            linear-gradient(135deg, rgba(30,30,80,0.85), rgba(15,15,40,0.92));
        border: 1px solid rgba(129,140,248,0.28);
        border-radius: 22px;
        padding: 30px 34px;
        margin: 4px 0 22px 0;
        box-shadow: 0 12px 48px rgba(0,0,0,0.35);
        animation: mbFadeUp .55s ease-out both;
    }
    .mb-hero-row { display:flex; align-items:center; gap:18px; flex-wrap:wrap; }
    .mb-logo {
        width: 60px; height: 60px; min-width:60px;
        border-radius: 18px;
        display:flex; align-items:center; justify-content:center;
        font-size: 32px;
        background: linear-gradient(135deg, #6366f1, #4f46e5);
        box-shadow: 0 8px 24px rgba(99,102,241,0.45);
    }
    .mb-hero-title {
        font-size: 2.5em; font-weight: 800; line-height: 1.05; margin: 0;
        background: linear-gradient(135deg, #c7d2fe, #818cf8, #6366f1);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .mb-hero-sub { color: #b9c0e0; font-size: 1.02em; margin: 4px 0 0 0; }

    /* ── Trust strip ──────────────────────────────────────────── */
    .mb-trust { display:flex; gap:10px; flex-wrap:wrap; margin-top:18px; }
    .mb-trust-item {
        display:inline-flex; align-items:center; gap:7px;
        background: rgba(99,102,241,0.12);
        border: 1px solid rgba(129,140,248,0.22);
        color: #cdd3f0; font-size: 0.82em; font-weight:500;
        padding: 6px 13px; border-radius: 999px;
    }

    /* ── Summary stat cards ───────────────────────────────────── */
    .mb-stat-grid {
        display:grid; grid-template-columns: repeat(4, 1fr);
        gap: 14px; margin: 6px 0 20px 0;
    }
    @media (max-width: 720px){ .mb-stat-grid { grid-template-columns: repeat(2, 1fr); } }
    .mb-stat {
        background: linear-gradient(160deg, rgba(30,30,75,0.75), rgba(18,18,52,0.85));
        border: 1px solid rgba(129,140,248,0.18);
        border-radius: 16px; padding: 16px 18px;
        transition: transform .18s ease, box-shadow .18s ease, border-color .18s ease;
        animation: mbFadeUp .5s ease-out both;
    }
    .mb-stat:hover { transform: translateY(-3px); box-shadow: 0 10px 28px rgba(0,0,0,0.35); }
    .mb-stat-num { font-size: 2.1em; font-weight: 800; line-height: 1; }
    .mb-stat-label { font-size: .82em; color: #aab1d6; margin-top: 6px; letter-spacing:.02em; }
    .mb-stat.total   { border-color: rgba(129,140,248,0.35); }
    .mb-stat.total   .mb-stat-num   { color: #a5b4fc; }
    .mb-stat.normal  .mb-stat-num   { color: #34d399; }
    .mb-stat.normal:hover  { border-color: rgba(52,211,153,0.5); }
    .mb-stat.border  .mb-stat-num   { color: #fbbf24; }
    .mb-stat.border:hover  { border-color: rgba(251,191,36,0.5); }
    .mb-stat.crit    .mb-stat-num   { color: #f87171; }
    .mb-stat.crit:hover    { border-color: rgba(248,113,113,0.55); }

    /* ── Empty states ─────────────────────────────────────────── */
    .mb-empty {
        text-align:center; padding: 40px 24px;
        border: 1px dashed rgba(129,140,248,0.28);
        border-radius: 18px;
        background: rgba(20,20,55,0.4);
        margin: 14px 0;
    }
    .mb-empty-icon { font-size: 2.6em; opacity:.9; }
    .mb-empty-title { font-size: 1.15em; font-weight:700; color:#e5e7eb; margin: 8px 0 4px; }
    .mb-empty-text { color:#9aa2c8; font-size: .92em; max-width: 440px; margin: 0 auto; }
    .mb-chips { display:flex; gap:8px; flex-wrap:wrap; justify-content:center; margin-top:16px; }
    .mb-chip {
        background: rgba(99,102,241,0.14);
        border: 1px solid rgba(129,140,248,0.25);
        color:#c7cdf0; font-size:.82em; font-weight:500;
        padding: 5px 12px; border-radius: 999px;
    }

    /* ── Footer ───────────────────────────────────────────────── */
    .mb-footer {
        margin: 40px 0 6px; padding-top: 18px;
        border-top: 1px solid rgba(129,140,248,0.15);
        text-align:center; color:#8088ad; font-size: .82em; line-height:1.7;
    }
    .mb-footer b { color:#aab1d6; }

    /* ── Card hover / entrance ────────────────────────────────── */
    .summary-card, .result-card { animation: mbFadeUp .5s ease-out both; }
    .result-card { transition: transform .18s ease, box-shadow .18s ease, border-color .18s ease; }
    .result-card:hover {
        transform: translateY(-2px);
        border-color: rgba(129,140,248,0.4);
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    }

    /* ── Buttons ──────────────────────────────────────────────── */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1, #4f46e5);
        color: #fff; border: none; border-radius: 12px;
        font-weight: 600; padding: 0.55rem 1rem;
        transition: transform .15s ease, box-shadow .15s ease, filter .15s ease;
        box-shadow: 0 6px 18px rgba(79,70,229,0.35);
    }
    .stButton > button:hover {
        transform: translateY(-2px); filter: brightness(1.07);
        box-shadow: 0 10px 26px rgba(79,70,229,0.5);
    }
    .stButton > button:active { transform: translateY(0); }

    /* Inputs & uploader */
    .stTextInput input, .stChatInput textarea {
        border-radius: 12px !important;
        border: 1px solid rgba(129,140,248,0.25) !important;
    }
    section[data-testid="stFileUploaderDropzone"] {
        border: 1px dashed rgba(129,140,248,0.35) !important;
        border-radius: 16px !important;
        background: rgba(20,20,55,0.4) !important;
    }

    /* Chat bubbles refinement */
    .chat-user, .chat-bot { animation: mbFadeUp .35s ease-out both; line-height:1.55; }

    /* Tabs */
    button[data-baseweb="tab"] { font-weight: 600; }

    /* Scrollbar */
    ::-webkit-scrollbar { width: 10px; height: 10px; }
    ::-webkit-scrollbar-thumb { background: rgba(99,102,241,0.4); border-radius: 8px; }
    ::-webkit-scrollbar-thumb:hover { background: rgba(99,102,241,0.6); }

    @keyframes mbFadeUp {
        from { opacity: 0; transform: translateY(10px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    @media (prefers-reduced-motion: reduce) {
        *, .mb-hero, .mb-stat, .summary-card, .result-card, .chat-user, .chat-bot {
            animation: none !important; transition: none !important;
        }
    }
</style>
""", unsafe_allow_html=True)


# ─── Session State Initialization ──────────────────────────────────
if "conversation_memory" not in st.session_state:
    st.session_state.conversation_memory = ConversationMemory()
if "report_analyzed" not in st.session_state:
    st.session_state.report_analyzed = False
if "report_summary" not in st.session_state:
    st.session_state.report_summary = ""
if "report_entities" not in st.session_state:
    st.session_state.report_entities = []
if "report_method" not in st.session_state:
    st.session_state.report_method = ""
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []
if "report_text" not in st.session_state:
    st.session_state.report_text = ""


def get_flag_html(flag: str, labels: dict) -> str:
    """Return styled HTML for a flag badge."""
    flag_map = {
        "NORMAL": f'<span class="flag-normal">{labels.get("normal", "🟢 NORMAL")}</span>',
        "BORDERLINE": f'<span class="flag-borderline">{labels.get("borderline", "🟡 BORDERLINE")}</span>',
        "CRITICAL": f'<span class="flag-critical">{labels.get("critical", "🔴 CRITICAL")}</span>',
        "UNKNOWN": f'<span class="flag-unknown">{labels.get("unknown", "⚪ UNKNOWN")}</span>',
    }
    return flag_map.get(flag, flag_map["UNKNOWN"])


def generate_pdf_summary(summary: str, entities: list, labels: dict) -> bytes:
    """Generate a downloadable PDF summary using fpdf2."""
    try:
        from fpdf import FPDF

        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 18)
        pdf.cell(0, 12, "MedBuddy - Report Summary", new_x="LMARGIN", new_y="NEXT", align="C")
        pdf.ln(8)

        # Summary
        pdf.set_font("Helvetica", "", 11)
        # Handle encoding by replacing unsupported characters
        safe_summary = summary.encode('latin-1', errors='replace').decode('latin-1')
        pdf.multi_cell(0, 6, safe_summary)
        pdf.ln(8)

        # Entities table
        if entities:
            pdf.set_font("Helvetica", "B", 13)
            pdf.cell(0, 10, "Detailed Results", new_x="LMARGIN", new_y="NEXT")
            pdf.ln(4)

            # Table header
            pdf.set_font("Helvetica", "B", 9)
            col_widths = [45, 25, 20, 30, 70]
            headers = ["Test", "Value", "Unit", "Status", "Reference Range"]
            for i, header in enumerate(headers):
                pdf.cell(col_widths[i], 8, header, border=1)
            pdf.ln()

            # Table rows
            pdf.set_font("Helvetica", "", 9)
            for entity in entities:
                term = str(entity.get("term", ""))[:25]
                value = str(entity.get("patient_value", ""))
                unit = str(entity.get("unit", ""))
                flag = str(entity.get("flag", "UNKNOWN"))
                ref_range = str(entity.get("reference_range", ""))

                row_data = [term, value, unit, flag, ref_range]
                for i, data in enumerate(row_data):
                    safe_data = data.encode('latin-1', errors='replace').decode('latin-1')
                    pdf.cell(col_widths[i], 7, safe_data, border=1)
                pdf.ln()

        return bytes(pdf.output())

    except Exception as e:
        print(f"[PDF Generation] Error: {e}")
        return b""


# ─── SIDEBAR ───────────────────────────────────────────────────────
with st.sidebar:
    language = st.radio(
        "🌐 Language / भाषा",
        SUPPORTED_LANGUAGES,
        index=0,
        key="language_select"
    )

    labels = UI_LABELS.get(language, UI_LABELS["English"])

    st.markdown(f"# {labels['app_title']}")
    st.markdown(f"*{labels['app_subtitle']}*")
    st.divider()

    literacy_level = st.radio(
        labels["literacy_label"],
        list(LITERACY_LEVELS.keys()),
        index=1,
        key="literacy_select"
    )

    st.divider()

    # About section
    with st.expander(labels["about_title"]):
        st.markdown(labels["about_text"])
        st.markdown("---")
        st.markdown("**Tech Stack:**")
        st.markdown("• LLM: Groq API (LLaMA 3.3 + Mixtral)")
        st.markdown("• RAG: FAISS + MedlinePlus")
        st.markdown("• OCR: PyMuPDF + Tesseract")
        st.markdown("• Framework: LangChain + Streamlit")

    # API Key check
    if not GROQ_API_KEY or GROQ_API_KEY == "your_groq_api_key_here":
        st.error("⚠️ GROQ_API_KEY not set! Create a `.env` file with your API key.")


# ─── MAIN AREA ─────────────────────────────────────────────────────
labels = UI_LABELS.get(language, UI_LABELS["English"])

# ─── Hero header (presentation only) ───────────────────────────────
if language == "English":
    _trust_items = [
        ("🔒", "Private — processed in memory"),
        ("📚", "Grounded in MedlinePlus"),
        ("🧪", "RAG + hallucination check"),
        ("⚕️", "Explains, never diagnoses"),
    ]
else:
    _trust_items = [
        ("🔒", "निजी — मेमोरी में संसाधित"),
        ("📚", "MedlinePlus पर आधारित"),
        ("🧪", "RAG + तथ्य जाँच"),
        ("⚕️", "समझाता है, निदान नहीं"),
    ]
_trust_html = "".join(
    f'<span class="mb-trust-item">{_ic} {_tx}</span>' for _ic, _tx in _trust_items
)
st.markdown(f"""
<div class="mb-hero">
    <div class="mb-hero-row">
        <div class="mb-logo">🩺</div>
        <div>
            <p class="mb-hero-title">MedBuddy</p>
            <p class="mb-hero-sub">{labels['app_subtitle']}</p>
        </div>
    </div>
    <div class="mb-trust">{_trust_html}</div>
</div>
""", unsafe_allow_html=True)

tab1, tab2 = st.tabs([labels["tab1_title"], labels["tab2_title"]])


# ═══════════════════════════════════════════════════════════════════
# TAB 1: ASK A TERM (MODE 1)
# ═══════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### " + labels["tab1_title"])

    col1, col2 = st.columns([4, 1])
    with col1:
        term_input = st.text_input(
            "term_query",
            placeholder=labels["term_input_placeholder"],
            label_visibility="collapsed",
            key="term_input"
        )
    with col2:
        explain_clicked = st.button(labels["explain_button"], key="explain_btn", use_container_width=True)

    # Empty state with example terms (presentation only)
    if not (explain_clicked and term_input):
        _examples = ["Hemoglobin", "HbA1c", "eGFR", "LDL Cholesterol", "TSH", "Vitamin D", "Creatinine"]
        _chips = "".join(f'<span class="mb-chip">{_t}</span>' for _t in _examples)
        _et_title = "Ask about any lab term" if language == "English" else "किसी भी लैब शब्द के बारे में पूछें"
        _et_text = (
            "Type a test name above and MedBuddy explains it in plain language, grounded in MedlinePlus."
            if language == "English"
            else "ऊपर कोई जांच नाम लिखें — MedBuddy इसे सरल भाषा में समझाएगा।"
        )
        st.markdown(f"""
        <div class="mb-empty">
            <div class="mb-empty-icon">🔍</div>
            <div class="mb-empty-title">{_et_title}</div>
            <div class="mb-empty-text">{_et_text}</div>
            <div class="mb-chips">{_chips}</div>
        </div>
        """, unsafe_allow_html=True)

    if explain_clicked and term_input:
        with st.spinner("🔍 Analyzing..." if language == "English" else "🔍 विश्लेषण कर रहे हैं..."):
            # Step 1: Classify intent
            classification = classify(term_input)
            intent = classification["intent"]

            if intent == "OUT_OF_SCOPE":
                st.warning(handle_refusal(language))

            elif intent == "REPORT_QUESTION":
                st.info(handle_report_redirect(language))

            elif intent == "MEDICAL_TERM":
                normalized = classification.get("normalized_term", term_input)

                # Step 2: RAG retrieval
                rag_result = retrieve(normalized)

                # Step 3: Generate explanation
                explain_result = explain(
                    term=normalized,
                    rag_result=rag_result,
                    literacy_level=literacy_level,
                    language=language
                )

                # Step 4: Hallucination verification (Tier 1 & 2 only)
                verification = None
                if not explain_result.get("disclaimer", True) and rag_result.get("context"):
                    verification = verify(
                        context=rag_result["context"],
                        explanation=explain_result["explanation"]
                    )

                # ─── Display Result Card ────────────────────────────
                st.markdown(f"""
                <div class="summary-card">
                    <h3>📋 {normalized}</h3>
                    <hr style="border-color: rgba(100,100,255,0.2);">
                    <p>{explain_result['explanation']}</p>
                """, unsafe_allow_html=True)

                # Source & grounding badge
                source_display = rag_result.get("source", "LLM_fallback")
                if source_display == "FAISS":
                    source_display = "MedlinePlus (cached)"
                elif source_display == "MedlinePlus_live":
                    source_display = "MedlinePlus (live)"
                elif source_display == "LLM_fallback":
                    source_display = "AI Knowledge"

                badge_html = ""
                if verification and verification.get("verdict") == "PASS":
                    score = verification.get("grounding_score", 0)
                    badge_html = f'<span class="badge-verified">{labels["verified_label"]} ({score:.2f})</span>'
                elif explain_result.get("disclaimer"):
                    badge_html = f'<span class="badge-unverified">{labels["unverified_label"]}</span>'
                else:
                    score = verification.get("grounding_score", 0) if verification else 0
                    badge_html = f'<span class="badge-unverified">⚠️ FLAG ({score:.2f})</span>'

                st.markdown(f"""
                    <p><strong>{labels['source_label']}:</strong> {source_display} &nbsp; {badge_html}</p>
                </div>
                """, unsafe_allow_html=True)

                # Show disclaimer for Tier 3
                if explain_result.get("disclaimer"):
                    disclaimer_text = (
                        "⚠️ This explanation is based on general AI knowledge and has not been verified against a certified medical source."
                        if language == "English"
                        else "⚠️ यह व्याख्या सामान्य AI ज्ञान पर आधारित है और किसी प्रमाणित चिकित्सा स्रोत से सत्यापित नहीं है।"
                    )
                    st.markdown(f'<div class="disclaimer-box">{disclaimer_text}</div>', unsafe_allow_html=True)

                # Source URL link
                if rag_result.get("source_url"):
                    st.markdown(f"🔗 [MedlinePlus Source]({rag_result['source_url']})")


# ═══════════════════════════════════════════════════════════════════
# TAB 2: UPLOAD REPORT (MODE 2)
# ═══════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### " + labels["tab2_title"])

    uploaded_file = st.file_uploader(
        labels["upload_label"],
        type=["pdf"],
        key="pdf_uploader"
    )

    analyze_clicked = st.button(labels["analyze_button"], key="analyze_btn", use_container_width=True)

    if analyze_clicked and uploaded_file:
        # Clear previous report session
        st.session_state.report_analyzed = False
        st.session_state.chat_messages = []
        st.session_state.conversation_memory.clear()

        with st.spinner(labels["analyzing_msg"]):
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            try:
                # ─── STEP 1: Document Intelligence ──────────────────
                doc_result = extract(tmp_path)

                if not doc_result.get("success", False) or not doc_result.get("text", "").strip():
                    st.error(
                        "❌ Could not extract text from this PDF. Possible reasons:\n"
                        "- The PDF is a scanned image (needs OCR dependencies: Tesseract + Poppler)\n"
                        "- The PDF is encrypted or password-protected\n"
                        "- The file is corrupted\n\n"
                        "**Fix:** Install Tesseract and Poppler, or try uploading a digital PDF."
                    )
                    st.stop()

                report_text = doc_result["text"]
                extraction_method = doc_result["method"]
                st.session_state.report_text = report_text
                st.session_state.report_method = extraction_method

                # Show extraction method badge
                if extraction_method == "digital":
                    st.markdown(f'<span class="badge-digital">{labels["digital_badge"]}</span>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<span class="badge-ocr">{labels["ocr_badge"]}</span>', unsafe_allow_html=True)

                # ─── STEP 2: Report Analysis ────────────────────────
                analysis = analyze(report_text)

                # ─── STEP 3: Report Summary (Prompt A) ──────────────
                summary = summarize(
                    structured_report_text=report_text,
                    report_analyzer_output=analysis,
                    literacy_level=literacy_level,
                    language=language
                )
                st.session_state.report_summary = summary

                # ─── STEP 4: Entity Extraction (Prompt B) ───────────
                entities = extract_entities(report_text)

                # ─── STEP 5: Value Contextualization ────────────────
                entities = add_flags_to_entities(entities)

                # ─── STEP 6: RAG Explanation for each entity ────────
                for entity in entities:
                    rag_result = retrieve(entity["term"])
                    explain_result = explain(
                        term=entity["term"],
                        value=entity.get("patient_value", ""),
                        unit=entity.get("unit", ""),
                        ref_range=entity.get("reference_range", ""),
                        flag=entity.get("flag", ""),
                        rag_result=rag_result,
                        literacy_level=literacy_level,
                        language=language
                    )
                    entity["explanation"] = explain_result["explanation"]
                    entity["source"] = explain_result["source"]
                    entity["disclaimer"] = explain_result.get("disclaimer", False)

                    # Hallucination verification for Tier 1 & 2
                    if not explain_result.get("disclaimer") and rag_result.get("context"):
                        verification = verify(rag_result["context"], explain_result["explanation"])
                        entity["grounding_score"] = verification.get("grounding_score")
                        entity["verification_verdict"] = verification.get("verdict")
                    else:
                        entity["grounding_score"] = None
                        entity["verification_verdict"] = None

                st.session_state.report_entities = entities

                # ─── STEP 7: Initialize Conversation Memory ────────
                st.session_state.conversation_memory.set_report_context(
                    narrative_summary=summary,
                    entities_with_flags=entities
                )

                st.session_state.report_analyzed = True

            except Exception as e:
                st.error(f"Error analyzing report: {str(e)}")
                import traceback
                traceback.print_exc()
            finally:
                # Clean up temp file
                try:
                    os.unlink(tmp_path)
                except:
                    pass

    # Nudge + empty state (presentation only)
    if analyze_clicked and not uploaded_file:
        st.warning(
            "⚠️ Please upload a PDF lab report first." if language == "English"
            else "⚠️ कृपया पहले एक PDF रिपोर्ट अपलोड करें।"
        )
    elif not st.session_state.report_analyzed and not (analyze_clicked and uploaded_file):
        if language == "English":
            _u_title = "Upload a lab report to begin"
            _u_text = "MedBuddy reads digital or scanned PDFs, flags each value, and explains your results — then answers your questions."
            _u_steps = ["1 · Upload PDF", "2 · Instant summary", "3 · Ask questions"]
        else:
            _u_title = "शुरू करने के लिए रिपोर्ट अपलोड करें"
            _u_text = "MedBuddy डिजिटल या स्कैन की गई PDF पढ़ता है, हर मान को फ़्लैग करता है और आपके परिणाम समझाता है।"
            _u_steps = ["1 · अपलोड", "2 · सारांश", "3 · सवाल पूछें"]
        _u_chips = "".join(f'<span class="mb-chip">{_s}</span>' for _s in _u_steps)
        st.markdown(f"""
        <div class="mb-empty">
            <div class="mb-empty-icon">📄</div>
            <div class="mb-empty-title">{_u_title}</div>
            <div class="mb-empty-text">{_u_text}</div>
            <div class="mb-chips">{_u_chips}</div>
        </div>
        """, unsafe_allow_html=True)

    # ─── Display Results (persisted in session state) ──────────────
    if st.session_state.report_analyzed:
        # Method badge
        method = st.session_state.report_method
        if method == "digital":
            st.markdown(f'<span class="badge-digital">{labels["digital_badge"]}</span>', unsafe_allow_html=True)
        elif method in ("ocr", "hybrid"):
            st.markdown(f'<span class="badge-ocr">{labels["ocr_badge"]}</span>', unsafe_allow_html=True)

        # ─── At-a-glance stats (derived from existing flags, no new logic) ───
        _ents = st.session_state.report_entities or []
        if _ents:
            _flags = [e.get("flag", "UNKNOWN") for e in _ents]
            _n_total = len(_flags)
            _n_norm = _flags.count("NORMAL")
            _n_bord = _flags.count("BORDERLINE")
            _n_crit = _flags.count("CRITICAL")
            if language == "English":
                _sl = ("Tests found", "Normal", "Borderline", "Needs attention")
            else:
                _sl = ("कुल जांच", "सामान्य", "सीमारेखा", "ध्यान दें")
            st.markdown(f"""
            <div class="mb-stat-grid">
                <div class="mb-stat total"><div class="mb-stat-num">{_n_total}</div><div class="mb-stat-label">{_sl[0]}</div></div>
                <div class="mb-stat normal"><div class="mb-stat-num">{_n_norm}</div><div class="mb-stat-label">{_sl[1]}</div></div>
                <div class="mb-stat border"><div class="mb-stat-num">{_n_bord}</div><div class="mb-stat-label">{_sl[2]}</div></div>
                <div class="mb-stat crit"><div class="mb-stat-num">{_n_crit}</div><div class="mb-stat-label">{_sl[3]}</div></div>
            </div>
            """, unsafe_allow_html=True)

        # ─── Summary Card ──────────────────────────────────────
        st.markdown(f"""
        <div class="summary-card">
            <h3>{labels['summary_header']}</h3>
            <hr style="border-color: rgba(100,100,255,0.2);">
            <p>{st.session_state.report_summary}</p>
        </div>
        """, unsafe_allow_html=True)

        # ─── Detailed Results Table ────────────────────────────
        entities = st.session_state.report_entities
        if entities:
            st.markdown(f"### {labels['results_header']}")

            for entity in entities:
                flag = entity.get("flag", "UNKNOWN")
                flag_html = get_flag_html(flag, labels)

                # Grounding badge
                grounding_html = ""
                if entity.get("grounding_score") is not None:
                    score = entity["grounding_score"]
                    verdict = entity.get("verification_verdict", "FLAG")
                    if verdict == "PASS":
                        grounding_html = f'<span class="badge-verified">{labels["verified_label"]} ({score:.2f})</span>'
                    else:
                        grounding_html = f'<span class="badge-unverified">⚠️ FLAG ({score:.2f})</span>'
                elif entity.get("disclaimer"):
                    grounding_html = f'<span class="badge-unverified">{labels["unverified_label"]}</span>'

                st.markdown(f"""
                <div class="result-card">
                    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
                        <h4 style="margin: 0;">{entity.get('term', 'Unknown')}</h4>
                        <div>{flag_html} &nbsp; {grounding_html}</div>
                    </div>
                    <div style="display: flex; gap: 20px; margin: 8px 0; flex-wrap: wrap;">
                        <span><strong>{labels['col_value']}:</strong> {entity.get('patient_value', 'N/A')}</span>
                        <span><strong>{labels['col_unit']}:</strong> {entity.get('unit', '')}</span>
                        <span><strong>Ref:</strong> {entity.get('reference_range', 'N/A')}</span>
                    </div>
                    <div class="explanation-card">
                        {entity.get('explanation', '')}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # ─── Download Button ───────────────────────────────
            pdf_bytes = generate_pdf_summary(
                st.session_state.report_summary,
                entities,
                labels
            )
            if pdf_bytes:
                st.download_button(
                    labels["download_button"],
                    data=pdf_bytes,
                    file_name="MedBuddy_Report_Summary.pdf",
                    mime="application/pdf",
                    key="download_pdf"
                )

        # ─── Conversational Q&A ────────────────────────────────
        st.divider()
        st.markdown(f"### {labels['chat_header']}")

        # Display chat history
        for msg in st.session_state.chat_messages:
            if msg["role"] == "user":
                st.markdown(f'<div class="chat-user">🧑 {msg["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-bot">🩺 {msg["content"]}</div>', unsafe_allow_html=True)

        # Chat input
        user_question = st.chat_input(labels["chat_placeholder"])
        if user_question:
            # Add user message
            st.session_state.chat_messages.append({"role": "user", "content": user_question})

            # Get response from conversation memory
            response = st.session_state.conversation_memory.ask(
                user_question=user_question,
                language=language
            )

            # Add bot response
            st.session_state.chat_messages.append({"role": "assistant", "content": response})

            # Rerun to show the new messages
            st.rerun()


# ─── Footer (presentation only) ────────────────────────────────────
if language == "English":
    _foot = (
        "<b>MedBuddy</b> — understand your health in plain language &nbsp;·&nbsp; "
        "Research-backed methodology (IEEE ESCI 2026)<br>"
        "Educational use only — MedBuddy explains lab results and does <b>not</b> diagnose or "
        "prescribe. Always consult a qualified doctor."
    )
else:
    _foot = (
        "<b>MedBuddy</b> — अपनी सेहत को सरल भाषा में समझें<br>"
        "केवल शैक्षिक उपयोग — MedBuddy निदान या उपचार नहीं देता। कृपया किसी योग्य डॉक्टर से सलाह लें।"
    )
st.markdown(f'<div class="mb-footer">{_foot}</div>', unsafe_allow_html=True)
