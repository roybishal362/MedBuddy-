"""
MedBuddy — Document Intelligence Module
4-step adaptive pipeline: PyMuPDF → OCR fallback → Table extraction → Structured output
"""
import os
import re
import fitz  # PyMuPDF
import pdfplumber
import numpy as np
from PIL import Image

# Optional OCR imports — gracefully handle missing system dependencies
try:
    from pdf2image import convert_from_path
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False

try:
    import pytesseract
    PYTESSERACT_AVAILABLE = True
except ImportError:
    PYTESSERACT_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config.settings import OCR_TEXT_MIN_LENGTH, OCR_DPI


def pymupdf_extract(pdf_path: str) -> str:
    """Extract text from PDF using PyMuPDF (works for digital PDFs)."""
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text("text") + "\n"
    doc.close()
    return text.strip()


def pdfplumber_extract_tables(pdf_path: str) -> list:
    """Extract tables from digital PDF using pdfplumber."""
    tables = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                page_tables = page.extract_tables()
                if page_tables:
                    for table in page_tables:
                        if table:
                            tables.append({
                                "page": page_num + 1,
                                "rows": table
                            })
    except Exception as e:
        print(f"[DocumentIntelligence] pdfplumber table extraction warning: {e}")
    return tables


def normalize_table_rows(tables: list) -> str:
    """Convert table data into normalized text format for LLM consumption."""
    normalized_lines = []
    for table_data in tables:
        rows = table_data.get("rows", [])
        if not rows:
            continue
        # Try to identify header row
        header = rows[0] if rows else []
        for row in rows[1:]:
            if not row or all(cell is None or str(cell).strip() == "" for cell in row):
                continue
            # Build normalized line
            parts = []
            for i, cell in enumerate(row):
                cell_text = str(cell).strip() if cell else ""
                if cell_text:
                    header_label = str(header[i]).strip() if i < len(header) and header[i] else f"Col{i}"
                    parts.append(f"{header_label}: {cell_text}")
            if parts:
                normalized_lines.append(" | ".join(parts))
    return "\n".join(normalized_lines)


def extract_tables_from_ocr_text(ocr_text: str) -> list:
    """Extract table-like data from OCR text using regex patterns."""
    tables = []
    # Pattern: Term followed by value, unit, and reference range
    patterns = [
        # Pattern 1: Test Name ... Value Unit Reference
        r'([A-Za-z\s\(\)]+?)\s+(\d+\.?\d*)\s*(g/dL|mg/dL|K/μL|%|mmol/L|U/L|mIU/L|ng/mL|pg/mL|fL|10\^3/μL|10\^6/μL|mm/hr|IU/mL|μg/dL|mEq/L|mg/L|ng/dL|pmol/L|nmol/L|cells/μL|copies/mL)?\s*[\-–]?\s*(?:(\d+\.?\d*)\s*[\-–]\s*(\d+\.?\d*))?',
        # Pattern 2: Simpler format
        r'([A-Za-z][A-Za-z\s\-]+)\s*[:\|]\s*(\d+\.?\d*)\s*([A-Za-z/μ%]+)?\s*(?:[:\|]\s*(\d+\.?\d*)\s*[\-–]\s*(\d+\.?\d*))?',
    ]

    lines = ocr_text.split('\n')
    for line in lines:
        line = line.strip()
        if not line:
            continue
        for pattern in patterns:
            match = re.match(pattern, line)
            if match:
                groups = match.groups()
                term = groups[0].strip() if groups[0] else ""
                value = groups[1] if len(groups) > 1 else ""
                unit = groups[2] if len(groups) > 2 and groups[2] else ""
                ref_low = groups[3] if len(groups) > 3 and groups[3] else ""
                ref_high = groups[4] if len(groups) > 4 and groups[4] else ""

                if term and value and len(term) > 2:
                    ref_range = f"{ref_low}-{ref_high}" if ref_low and ref_high else ""
                    tables.append({
                        "term": term,
                        "value": value,
                        "unit": unit,
                        "reference_range": ref_range
                    })
                break

    return tables


def preprocess_image_for_ocr(img: Image.Image) -> np.ndarray:
    """Apply OpenCV preprocessing to improve OCR accuracy."""
    if not CV2_AVAILABLE:
        return np.array(img)

    img_array = np.array(img)
    # Convert to grayscale
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array

    # Denoise
    denoised = cv2.fastNlMeansDenoising(gray, h=10)

    # Otsu's thresholding
    thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    return thresh


def merge_text_and_tables(text: str, tables: list) -> str:
    """Merge extracted text with normalized table data."""
    normalized = normalize_table_rows(tables)
    if normalized:
        return f"{text}\n\n--- Extracted Table Data ---\n{normalized}"
    return text


def extract(pdf_path: str) -> dict:
    """
    Main extraction function — 4-step adaptive pipeline.

    Returns:
      {
        "text": str,           # full cleaned structured text
        "method": str,         # "digital" | "ocr" | "hybrid"
        "tables": list[dict],  # structured tables if detected
        "page_count": int,
        "success": bool        # whether extraction was successful
      }
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    # Count pages
    page_count = 0
    try:
        doc = fitz.open(pdf_path)
        page_count = len(doc)
        doc.close()
    except Exception as e:
        print(f"[DocumentIntelligence] Unable to get page count: {e}")

    # STEP 1: Digital extraction via PyMuPDF
    text = ""
    try:
        text = pymupdf_extract(pdf_path)
    except Exception as e:
        print(f"[DocumentIntelligence] PyMuPDF extraction error: {e}")

    if len(text.strip()) >= OCR_TEXT_MIN_LENGTH:
        # STEP 2a: Digital PDF — extract tables with pdfplumber
        tables = pdfplumber_extract_tables(pdf_path)
        structured = merge_text_and_tables(text, tables)
        return {
            "text": structured,
            "method": "digital",
            "tables": tables,
            "page_count": page_count,
            "success": True
        }

    # STEP 2b: Scanned PDF — try OCR if available
    if not PDF2IMAGE_AVAILABLE or not PYTESSERACT_AVAILABLE:
        # Return whatever text we got, even if short
        if text.strip():
            return {
                "text": text,
                "method": "digital",
                "tables": [],
                "page_count": page_count,
                "success": True
            }
        return {
            "text": "",
            "method": "digital",
            "tables": [],
            "page_count": page_count,
            "success": False
        }

    try:
        images = convert_from_path(pdf_path, dpi=OCR_DPI)
    except Exception as e:
        print(f"[DocumentIntelligence] PDF to image conversion failed: {e}")
        # Still return any digital text we managed to get
        if text.strip():
            return {
                "text": text,
                "method": "digital",
                "tables": [],
                "page_count": page_count,
                "success": True
            }
        return {
            "text": "",
            "method": "ocr",
            "tables": [],
            "page_count": page_count,
            "success": False
        }

    ocr_text = ""
    for img in images:
        # STEP 3: OpenCV preprocessing for OCR accuracy
        processed = preprocess_image_for_ocr(img)

        # STEP 4: Tesseract OCR
        try:
            ocr_text += pytesseract.image_to_string(
                processed, lang="eng+hin",
                config="--psm 6"  # assume uniform block of text
            ) + "\n"
        except Exception:
            # Fallback to English only if Hindi pack not available
            try:
                ocr_text += pytesseract.image_to_string(
                    processed, lang="eng",
                    config="--psm 6"
                ) + "\n"
            except Exception as e2:
                print(f"[DocumentIntelligence] OCR Error: {e2}")

    # Combine best text
    final_text = ocr_text.strip() if ocr_text.strip() else text.strip()

    if not final_text:
        return {
            "text": "",
            "method": "ocr",
            "tables": [],
            "page_count": page_count,
            "success": False
        }

    # Extract tables from OCR text using regex
    ocr_tables = extract_tables_from_ocr_text(final_text)

    # Determine method
    method = "ocr" if not text else "hybrid"

    return {
        "text": final_text,
        "method": method,
        "tables": ocr_tables,
        "page_count": page_count,
        "success": True
    }

