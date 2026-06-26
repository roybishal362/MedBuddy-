"""
Generate sample lab report PDFs for MedBuddy demo and testing.
Run this script once to create 3 sample report PDFs.
"""
from fpdf import FPDF
import os

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "assets", "sample_reports")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def create_digital_report():
    """Create a clean digital lab report PDF."""
    pdf = FPDF()
    pdf.add_page()

    # Header
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 12, "HealthCare Diagnostics Laboratory", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, "123 Medical Center Drive, New Delhi, India", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.cell(0, 6, "Phone: +91-11-2345-6789 | NABL Accredited", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(8)

    # Patient Info
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "LABORATORY REPORT", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(4)

    pdf.set_font("Helvetica", "", 10)
    info = [
        ("Patient Name:", "John Doe (Demo Patient)"),
        ("Age / Gender:", "45 Years / Male"),
        ("Sample ID:", "LAB-2024-00456"),
        ("Sample Date:", "2024-12-15"),
        ("Report Date:", "2024-12-16"),
        ("Referred By:", "Dr. Smith (General Physician)"),
    ]
    for label, value in info:
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(40, 6, label)
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 6, value, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(6)

    # Complete Blood Count
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "COMPLETE BLOOD COUNT (CBC)", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    # Table header
    pdf.set_font("Helvetica", "B", 9)
    col_widths = [55, 25, 25, 35, 45]
    headers = ["Test Name", "Result", "Unit", "Reference Range", "Status"]
    for i, h in enumerate(headers):
        pdf.cell(col_widths[i], 8, h, border=1, align="C")
    pdf.ln()

    # CBC Data
    pdf.set_font("Helvetica", "", 9)
    cbc_data = [
        ("Hemoglobin (Hb)", "11.2", "g/dL", "13.5 - 17.5", "LOW"),
        ("Total RBC Count", "4.2", "million/uL", "4.5 - 5.5", "LOW"),
        ("Hematocrit (HCT)", "34.5", "%", "40 - 54", "LOW"),
        ("MCV", "82.1", "fL", "80 - 100", "Normal"),
        ("MCH", "26.7", "pg", "27 - 33", "Normal"),
        ("MCHC", "32.5", "g/dL", "32 - 36", "Normal"),
        ("Total WBC Count", "7.2", "K/uL", "4.0 - 11.0", "Normal"),
        ("Neutrophils", "62", "%", "40 - 75", "Normal"),
        ("Lymphocytes", "30", "%", "20 - 45", "Normal"),
        ("Monocytes", "5", "%", "2 - 10", "Normal"),
        ("Eosinophils", "2", "%", "1 - 6", "Normal"),
        ("Basophils", "1", "%", "0 - 2", "Normal"),
        ("Platelet Count", "250", "K/uL", "150 - 400", "Normal"),
        ("RDW", "14.2", "%", "11.5 - 14.5", "Normal"),
    ]
    for row in cbc_data:
        for i, cell in enumerate(row):
            pdf.cell(col_widths[i], 7, cell, border=1, align="C")
        pdf.ln()
    pdf.ln(6)

    # Blood Sugar
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "BLOOD SUGAR PROFILE", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    pdf.set_font("Helvetica", "B", 9)
    for i, h in enumerate(headers):
        pdf.cell(col_widths[i], 8, h, border=1, align="C")
    pdf.ln()

    pdf.set_font("Helvetica", "", 9)
    sugar_data = [
        ("Fasting Blood Sugar", "108", "mg/dL", "70 - 100", "HIGH"),
        ("HbA1c", "6.2", "%", "4.0 - 5.7", "HIGH"),
    ]
    for row in sugar_data:
        for i, cell in enumerate(row):
            pdf.cell(col_widths[i], 7, cell, border=1, align="C")
        pdf.ln()
    pdf.ln(6)

    # Lipid Profile
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "LIPID PROFILE", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    pdf.set_font("Helvetica", "B", 9)
    for i, h in enumerate(headers):
        pdf.cell(col_widths[i], 8, h, border=1, align="C")
    pdf.ln()

    pdf.set_font("Helvetica", "", 9)
    lipid_data = [
        ("Total Cholesterol", "215", "mg/dL", "< 200", "HIGH"),
        ("LDL Cholesterol", "142", "mg/dL", "< 100", "HIGH"),
        ("HDL Cholesterol", "42", "mg/dL", "> 40", "Normal"),
        ("Triglycerides", "155", "mg/dL", "< 150", "HIGH"),
        ("VLDL Cholesterol", "31", "mg/dL", "< 30", "HIGH"),
    ]
    for row in lipid_data:
        for i, cell in enumerate(row):
            pdf.cell(col_widths[i], 7, cell, border=1, align="C")
        pdf.ln()
    pdf.ln(6)

    # Kidney Function
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "KIDNEY FUNCTION TEST", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    pdf.set_font("Helvetica", "B", 9)
    for i, h in enumerate(headers):
        pdf.cell(col_widths[i], 8, h, border=1, align="C")
    pdf.ln()

    pdf.set_font("Helvetica", "", 9)
    kidney_data = [
        ("Blood Urea Nitrogen", "18", "mg/dL", "7 - 20", "Normal"),
        ("Creatinine", "0.9", "mg/dL", "0.7 - 1.3", "Normal"),
        ("Uric Acid", "6.2", "mg/dL", "3.5 - 7.2", "Normal"),
        ("eGFR", "95", "mL/min", "> 90", "Normal"),
    ]
    for row in kidney_data:
        for i, cell in enumerate(row):
            pdf.cell(col_widths[i], 7, cell, border=1, align="C")
        pdf.ln()

    # Footer
    pdf.ln(10)
    pdf.set_font("Helvetica", "I", 8)
    pdf.cell(0, 6, "* This is a demo report for MedBuddy testing. Not for clinical use.", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.cell(0, 6, "Report generated by HealthCare Diagnostics Laboratory", new_x="LMARGIN", new_y="NEXT", align="C")

    output_path = os.path.join(OUTPUT_DIR, "digital_report.pdf")
    pdf.output(output_path)
    print(f"Created: {output_path}")


def create_complex_table_report():
    """Create a report with complex table layout."""
    pdf = FPDF()
    pdf.add_page()

    # Header
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Metro City Pathology Lab", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.set_font("Helvetica", "", 9)
    pdf.cell(0, 5, "456 Hospital Road, Mumbai, Maharashtra 400001", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(6)

    # Patient Info in two columns
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "COMPREHENSIVE HEALTH CHECK REPORT", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(3)

    pdf.set_font("Helvetica", "", 9)
    pdf.cell(95, 5, "Patient: Priya Sharma (Demo)")
    pdf.cell(95, 5, "Age: 35 Years / Female", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(95, 5, "Sample ID: MCP-2024-789")
    pdf.cell(95, 5, "Date: 2024-12-20", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(6)

    # Thyroid Function Test
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "THYROID FUNCTION TEST", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    col_widths = [50, 25, 25, 40, 45]
    headers = ["Test", "Result", "Unit", "Reference Range", "Interpretation"]

    pdf.set_font("Helvetica", "B", 8)
    for i, h in enumerate(headers):
        pdf.cell(col_widths[i], 7, h, border=1, align="C")
    pdf.ln()

    pdf.set_font("Helvetica", "", 8)
    thyroid_data = [
        ("TSH", "8.5", "mIU/L", "0.4 - 4.0", "HIGH"),
        ("Free T3", "2.1", "pg/mL", "2.0 - 4.4", "Normal"),
        ("Free T4", "0.8", "ng/dL", "0.8 - 1.8", "Normal"),
        ("Total T3", "95", "ng/dL", "80 - 200", "Normal"),
        ("Total T4", "6.5", "ug/dL", "5.1 - 14.1", "Normal"),
        ("Anti-TPO Antibodies", "250", "IU/mL", "< 35", "HIGH"),
    ]
    for row in thyroid_data:
        for i, cell in enumerate(row):
            pdf.cell(col_widths[i], 6, cell, border=1, align="C")
        pdf.ln()
    pdf.ln(4)

    # Liver Function Test
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "LIVER FUNCTION TEST", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    pdf.set_font("Helvetica", "B", 8)
    for i, h in enumerate(headers):
        pdf.cell(col_widths[i], 7, h, border=1, align="C")
    pdf.ln()

    pdf.set_font("Helvetica", "", 8)
    liver_data = [
        ("AST (SGOT)", "85", "U/L", "10 - 40", "HIGH"),
        ("ALT (SGPT)", "92", "U/L", "7 - 56", "HIGH"),
        ("Alkaline Phosphatase", "110", "U/L", "44 - 147", "Normal"),
        ("GGT", "45", "U/L", "9 - 48", "Normal"),
        ("Total Bilirubin", "0.9", "mg/dL", "0.1 - 1.2", "Normal"),
        ("Direct Bilirubin", "0.2", "mg/dL", "0.0 - 0.3", "Normal"),
        ("Indirect Bilirubin", "0.7", "mg/dL", "0.1 - 1.0", "Normal"),
        ("Total Protein", "7.2", "g/dL", "6.0 - 8.3", "Normal"),
        ("Albumin", "4.1", "g/dL", "3.5 - 5.0", "Normal"),
        ("Globulin", "3.1", "g/dL", "2.0 - 3.5", "Normal"),
        ("A/G Ratio", "1.3", "", "1.0 - 2.5", "Normal"),
    ]
    for row in liver_data:
        for i, cell in enumerate(row):
            pdf.cell(col_widths[i], 6, cell, border=1, align="C")
        pdf.ln()
    pdf.ln(4)

    # Vitamin Panel
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "VITAMIN PANEL", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    pdf.set_font("Helvetica", "B", 8)
    for i, h in enumerate(headers):
        pdf.cell(col_widths[i], 7, h, border=1, align="C")
    pdf.ln()

    pdf.set_font("Helvetica", "", 8)
    vitamin_data = [
        ("Vitamin D (25-OH)", "18", "ng/mL", "30 - 100", "LOW"),
        ("Vitamin B12", "180", "pg/mL", "200 - 900", "LOW"),
        ("Folate", "8.5", "ng/mL", "3.0 - 17.0", "Normal"),
        ("Iron", "55", "ug/dL", "60 - 170", "LOW"),
        ("Ferritin", "12", "ng/mL", "20 - 200", "LOW"),
    ]
    for row in vitamin_data:
        for i, cell in enumerate(row):
            pdf.cell(col_widths[i], 6, cell, border=1, align="C")
        pdf.ln()

    # Footer
    pdf.ln(8)
    pdf.set_font("Helvetica", "I", 8)
    pdf.cell(0, 5, "* Demo report for MedBuddy testing. Not for clinical use.", new_x="LMARGIN", new_y="NEXT", align="C")

    output_path = os.path.join(OUTPUT_DIR, "complex_table_report.pdf")
    pdf.output(output_path)
    print(f"Created: {output_path}")


def create_scanned_report():
    """Create a simple report that simulates a scanned document (still digital but with less structure)."""
    pdf = FPDF()
    pdf.add_page()

    # Simulate a less structured "scanned" report
    pdf.set_font("Courier", "B", 14)
    pdf.cell(0, 10, "CITY HOSPITAL - LAB REPORT", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(4)

    pdf.set_font("Courier", "", 10)
    pdf.cell(0, 6, "Patient: Rahul Kumar (Demo)   Age: 55/M", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6, "Date: 15-Dec-2024   ID: CH-2024-1234", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(6)

    pdf.set_font("Courier", "B", 11)
    pdf.cell(0, 6, "BLOOD TEST RESULTS", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6, "=" * 55, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    pdf.set_font("Courier", "", 9)
    # Written in a more "scanned report" style
    results = [
        "Hemoglobin          14.5  g/dL        (13.5-17.5)   Normal",
        "RBC Count            5.2  million/uL  (4.5-5.5)     Normal",
        "WBC Count            6.8  K/uL        (4.0-11.0)    Normal",
        "Platelet Count     180    K/uL        (150-400)     Normal",
        "ESR                  35   mm/hr       (0-20)        HIGH",
        "",
        "Fasting Blood Sugar 145   mg/dL       (70-100)      HIGH",
        "PP Blood Sugar      210   mg/dL       (< 140)       HIGH",
        "HbA1c                7.8  %           (4.0-5.7)     HIGH",
        "",
        "Total Cholesterol   245   mg/dL       (< 200)       HIGH",
        "LDL Cholesterol     165   mg/dL       (< 100)       HIGH",
        "HDL Cholesterol      38   mg/dL       (> 40)        LOW",
        "Triglycerides       210   mg/dL       (< 150)       HIGH",
        "",
        "Creatinine           1.8  mg/dL       (0.7-1.3)     HIGH",
        "Blood Urea           45   mg/dL       (7-20)        HIGH",
        "Uric Acid            9.5  mg/dL       (3.5-7.2)     HIGH",
        "eGFR                 52   mL/min      (> 90)        LOW",
        "",
        "TSH                  2.5  mIU/L       (0.4-4.0)     Normal",
        "Calcium              9.4  mg/dL       (8.5-10.5)    Normal",
        "Sodium             140    mEq/L       (136-145)     Normal",
        "Potassium            4.2  mEq/L       (3.5-5.0)     Normal",
    ]

    for line in results:
        pdf.cell(0, 5, line, new_x="LMARGIN", new_y="NEXT")

    pdf.ln(8)
    pdf.set_font("Courier", "B", 9)
    pdf.cell(0, 6, "Pathologist: Dr. A. Gupta, MD (Pathology)", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)
    pdf.set_font("Courier", "I", 8)
    pdf.cell(0, 5, "* Demo report for MedBuddy testing purposes only.", new_x="LMARGIN", new_y="NEXT")

    output_path = os.path.join(OUTPUT_DIR, "scanned_report.pdf")
    pdf.output(output_path)
    print(f"Created: {output_path}")


if __name__ == "__main__":
    print("Generating sample lab report PDFs...")
    create_digital_report()
    create_complex_table_report()
    create_scanned_report()
    print("\nDone! All 3 sample reports created in:", OUTPUT_DIR)
