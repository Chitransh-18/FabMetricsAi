"""
Script to build a formal 10-Page Major Project Synopsis PDF for FabMetrics AI,
formatted according to Jaypee Institute of Information Technology (JIIT) guidelines.
"""

import os
import sys
import io
import time
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, HRFlowable, Image as RLImage
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

def create_synopsis_pdf(output_filename="FabMetrics_AI_JIIT_Synopsis.pdf"):
    pdf_path = Path(output_filename)
    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=letter,
        rightMargin=54,  # 0.75 in
        leftMargin=54,
        topMargin=54,
        bottomMargin=54,
        title="Major Project Synopsis — FabMetrics AI",
        author="FabMetrics AI Project Team",
        subject="JIIT Major Project Synopsis",
        creator="FabMetrics AI Generator"
    )

    styles = getSampleStyleSheet()

    # Custom Typography Styles matching JIIT format
    title_cover_style = ParagraphStyle(
        'CoverTitle', parent=styles['Normal'],
        fontName='Helvetica-Bold', fontSize=20, leading=24,
        alignment=1, textColor=colors.HexColor("#0f172a"), spaceAfter=15
    )
    center_text_style = ParagraphStyle(
        'CenterText', parent=styles['Normal'],
        fontName='Helvetica', fontSize=11, leading=16,
        alignment=1, textColor=colors.HexColor("#334155"), spaceAfter=10
    )
    bold_center_style = ParagraphStyle(
        'BoldCenterText', parent=styles['Normal'],
        fontName='Helvetica-Bold', fontSize=12, leading=16,
        alignment=1, textColor=colors.HexColor("#0f172a"), spaceAfter=10
    )
    
    heading1_style = ParagraphStyle(
        'Heading1Custom', parent=styles['Heading1'],
        fontName='Helvetica-Bold', fontSize=15, leading=19,
        alignment=1, textColor=colors.HexColor("#0f172a"),
        spaceBefore=15, spaceAfter=12
    )
    heading2_style = ParagraphStyle(
        'Heading2Custom', parent=styles['Heading2'],
        fontName='Helvetica-Bold', fontSize=11.5, leading=15,
        alignment=0, textColor=colors.HexColor("#1e293b"),
        spaceBefore=10, spaceAfter=6
    )
    body_style = ParagraphStyle(
        'BodyCustom', parent=styles['Normal'],
        fontName='Helvetica', fontSize=10, leading=14.5,
        alignment=4, textColor=colors.HexColor("#1e293b"), spaceAfter=8
    )
    bullet_style = ParagraphStyle(
        'BulletCustom', parent=styles['Normal'],
        fontName='Helvetica', fontSize=9.5, leading=14,
        leftIndent=15, textColor=colors.HexColor("#1e293b"), spaceAfter=4
    )
    quote_style = ParagraphStyle(
        'QuoteStyle', parent=styles['Normal'],
        fontName='Helvetica-Oblique', fontSize=11, leading=15,
        alignment=1, textColor=colors.HexColor("#475569"), spaceAfter=12
    )

    story = []

    # =========================================================================
    # PAGE 1: TITLE PAGE
    # =========================================================================
    story.append(Spacer(1, 15))
    story.append(Paragraph("A MAJOR PROJECT SYNOPSIS -", ParagraphStyle('P1', parent=center_text_style, fontName='Helvetica-Bold', fontSize=16, leading=20)))
    story.append(Paragraph("ON", ParagraphStyle('P2', parent=center_text_style, fontSize=12)))
    story.append(Paragraph("FABMETRICS AI: AUTOMATED SEMICONDUCTOR WAFER DEFECT INSPECTION & YIELD ENHANCEMENT PLATFORM", title_cover_style))
    story.append(Spacer(1, 10))
    story.append(Paragraph("SUBMITTED IN PARTIAL FULFILLMENT FOR THE AWARD OF DEGREE OF", center_text_style))
    story.append(Paragraph("BACHELOR OF TECHNOLOGY", ParagraphStyle('P3', parent=bold_center_style, fontSize=14)))
    story.append(Paragraph("IN", center_text_style))
    story.append(Paragraph("COMPUTER SCIENCE AND ENGINEERING / ELECTRONICS &amp; COMMUNICATION ENGINEERING", ParagraphStyle('P4', parent=bold_center_style, fontSize=11)))
    story.append(Spacer(1, 15))

    # Add Logo if present
    logo_path = Path("frontend/assets/logo.png")
    if logo_path.exists():
        try:
            story.append(RLImage(str(logo_path), width=85, height=85))
        except Exception:
            pass
    story.append(Spacer(1, 15))

    table_data = [
        [
            Paragraph("<b>Submitted by:</b>", body_style),
            Paragraph("<b>Under the Guidance of:</b>", body_style)
        ],
        [
            Paragraph("<b>CHITRANSH SAXENA</b> (Enrollment No.)<br/><b>CO-AUTHOR 1</b> (Enrollment No.)<br/><b>CO-AUTHOR 2</b> (Enrollment No.)", body_style),
            Paragraph("<b>DR. SUPERVISOR NAME</b><br/>Designation<br/>Department of CSE / ECE", body_style)
        ]
    ]
    t_sub = Table(table_data, colWidths=[240, 240])
    t_sub.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('LEFTPADDING', (0,0), (-1,-1), 0),
        ('RIGHTPADDING', (0,0), (-1,-1), 0),
    ]))
    story.append(t_sub)

    story.append(Spacer(1, 35))
    story.append(Paragraph("DEPARTMENT OF COMPUTER SCIENCE AND ENGINEERING / ECE", bold_center_style))
    story.append(Paragraph("JAYPEE INSTITUTE OF INFORMATION TECHNOLOGY, NOIDA (U.P.)", ParagraphStyle('P5', parent=bold_center_style, fontSize=11)))
    story.append(PageBreak())

    # =========================================================================
    # PAGE 2: CERTIFICATE PAGE
    # =========================================================================
    story.append(Paragraph("SEPTEMBER 2025 / FEBRUARY 2026", bold_center_style))
    story.append(Spacer(1, 15))
    story.append(Paragraph("CERTIFICATE", heading1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#0f172a"), spaceAfter=20))
    story.append(Spacer(1, 10))
    
    cert_text = (
        "This is to certify that the major project synopsis titled <b>“FabMetrics AI: Automated Semiconductor Wafer Defect Inspection &amp; Yield Enhancement Platform”</b> "
        "submitted by <b>Chitransh Saxena</b>, Co-Author 1, and Co-Author 2 is new, appropriate, and not repeated/copied from the previously submitted project works."
    )
    story.append(Paragraph(cert_text, body_style))
    story.append(Spacer(1, 120))

    story.append(Paragraph("<b>Signature of Supervisor:</b> ___________________________", body_style))
    story.append(Paragraph("<b>Name of the Supervisor:</b> Dr. Supervisor Name", body_style))
    story.append(Paragraph("Department of CSE / ECE,", body_style))
    story.append(Paragraph("JIIT, Sec-128 / Sec-62,", body_style))
    story.append(Paragraph("Noida-201304", body_style))
    story.append(Spacer(1, 20))
    story.append(Paragraph(f"<b>Dated:</b> {time.strftime('%d-%m-%Y')}", body_style))
    story.append(PageBreak())

    # =========================================================================
    # PAGE 3: DECLARATION PAGE
    # =========================================================================
    story.append(Paragraph("DECLARATION", heading1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#0f172a"), spaceAfter=20))
    story.append(Spacer(1, 10))

    decl_text = (
        "We hereby declare that the <b>“FABMETRICS AI”</b> Major Project submission is not repeated/copied from previously "
        "submitted project works and has not misrepresented, fabricated, or falsified any idea, dataset, fact, or source in our submission."
    )
    story.append(Paragraph(decl_text, body_style))
    story.append(Spacer(1, 100))

    story.append(Paragraph("<b>Place:</b> Noida", body_style))
    story.append(Paragraph(f"<b>Date:</b> {time.strftime('%d-%b-%Y')}", body_style))
    story.append(Spacer(1, 30))

    dec_table_data = [
        [Paragraph("<b>Name:</b> Chitransh Saxena", body_style), Paragraph("<b>Enrollment:</b> Enrollment No.", body_style)],
        [Paragraph("<b>Name:</b> Co-Author 1", body_style), Paragraph("<b>Enrollment:</b> Enrollment No.", body_style)],
        [Paragraph("<b>Name:</b> Co-Author 2", body_style), Paragraph("<b>Enrollment:</b> Enrollment No.", body_style)]
    ]
    t_dec = Table(dec_table_data, colWidths=[240, 240])
    t_dec.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 12),
    ]))
    story.append(t_dec)
    story.append(PageBreak())

    # =========================================================================
    # PAGE 4: PROBLEM STATEMENT
    # =========================================================================
    story.append(Paragraph("PROBLEM STATEMENT", heading1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#0f172a"), spaceAfter=15))
    story.append(Paragraph("<i>“Precision at the nanoscale determines survival in modern semiconductor fabrication.”</i>", quote_style))

    ps_1 = (
        "Semiconductor fabrication cleanrooms operate at extreme sub-micron scales where microscopic physical defects "
        "on silicon wafers can instantly destroy entire microchip yield batches. Conventional manual optical inspection (AOI) "
        "methods suffer from human fatigue, slow processing rates (exceeding 140ms per substrate), severe class imbalance, and "
        "high subjective error rates. Existing machine learning baselines (Radon+SVM, Shallow CNNs) fail to achieve the required "
        "reliability (&gt;95% Macro-F1) across complex overlapping failure modes."
    )
    story.append(Paragraph(ps_1, body_style))

    ps_2 = (
        "The <b>FabMetrics AI</b> project targets this industrial gap by constructing an automated, high-precision deep learning platform "
        "capable of real-time multi-class wafer defect classification, computer vision spatial localization, and automated yield report generation. "
        "The primary goal is to provide cleanroom yield engineers with an instant, sub-16ms diagnostic engine that identifies defect patterns "
        "(Scratch, Donut, Edge-Ring, Edge-Loc, Loc, Center, Near-full, Random, None, Multi-Defect), highlights defect die clusters using "
        "bounding box contours, and stores historical audit records under secure user-authenticated profiles."
    )
    story.append(Paragraph(ps_2, body_style))

    ps_3 = (
        "The platform incorporates a novel <b>Dual-Branch Cross-Attention Architecture (ResNet50-CBAM + EfficientNet-B0)</b> trained on 35,000 "
        "equalized wafer maps utilizing Focal Loss (&gamma;=2.0) and Stochastic Weight Averaging (SWA). It couples high-precision inference "
        "with an executive 4-page PDF yield audit generator and a cleanroom AI tutor chatbot, establishing a scalable solution for modern Industry 4.0 fabrication plants."
    )
    story.append(Paragraph(ps_3, body_style))
    story.append(PageBreak())

    # =========================================================================
    # PAGE 5: OBJECTIVES
    # =========================================================================
    story.append(Paragraph("OBJECTIVES", heading1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#0f172a"), spaceAfter=15))

    obj_intro = (
        "The purpose of the FabMetrics AI project is to create an integrated industrial computer vision platform that automates silicon wafer defect "
        "recognition and cleanroom yield analytics. By combining dual-stream deep neural networks with computer vision contour segmentation, "
        "FastAPI REST microservices, and interactive web dashboard interfaces, the system achieves sub-16ms automated inspection."
    )
    story.append(Paragraph(obj_intro, body_style))
    story.append(Spacer(1, 6))

    story.append(Paragraph("Dataset Acquisition &amp; Preprocessing Objectives", heading2_style))
    story.append(Paragraph("• To curate and preprocess 35,000 equalized wafer map samples from the benchmark WM-811K semiconductor dataset.", bullet_style))
    story.append(Paragraph("• To implement advanced spatial data augmentation (random rotations, axial flips, SMOTE balancing) to resolve severe class imbalance across 10 defect modes.", bullet_style))
    story.append(Paragraph("• To extract normalized 2D die matrix representations suitable for spatial and textural feature extraction pipelines.", bullet_style))

    story.append(Paragraph("Software / Deep Learning Algorithm Objectives", heading2_style))
    story.append(Paragraph("• To architect a Dual-Branch Cross-Attention Neural Network combining ResNet50-CBAM (spatial topology branch) and EfficientNet-B0 (surface texture branch).", bullet_style))
    story.append(Paragraph("• To optimize training convergence using Focal Loss (&gamma;=2.0) and Stochastic Weight Averaging (SWA) over 50 epochs.", bullet_style))
    story.append(Paragraph("• To achieve a state-of-the-art Validation Macro F1-score exceeding 97.5% at sub-16ms inference latency.", bullet_style))
    story.append(Paragraph("• To develop an automated Computer Vision contour isolation engine (OpenCV) to localize and bound defect die clusters with exact bounding box coordinates.", bullet_style))

    story.append(Paragraph("Backend Infrastructure &amp; Security Objectives", heading2_style))
    story.append(Paragraph("• To design a high-throughput FastAPI REST microservice supporting asynchronous batch wafer map processing.", bullet_style))
    story.append(Paragraph("• To implement secure SQLite database integration (`fabmetrics.db`) featuring salted PBKDF2 password hashing (100,000 iterations) and persistent Bearer session tokens.", bullet_style))
    story.append(Paragraph("• To enforce streaming file upload size limits (10MB) and SQLite Write-Ahead Logging (WAL) for lock-free multi-user concurrency.", bullet_style))

    story.append(Paragraph("User Interface &amp; Executive Reporting Objectives", heading2_style))
    story.append(Paragraph("• To construct an interactive Glassmorphic Web Dashboard (`frontend/index.html`) featuring 55 custom themes, dynamic contrast engine, and a 50-sample paginated visual showroom catalog.", bullet_style))
    story.append(Paragraph("• To build an automated 4-Page Executive PDF Yield Audit Report engine (ReportLab) featuring background patent watermarks (`REG US-2026-FABMETRICS-AI`).", bullet_style))
    story.append(Paragraph("• To integrate an embedded Cleanroom AI Tutor Chatbot with domain-specific guardrails for cleanroom failure physics and ISO 14644-1 standards.", bullet_style))
    story.append(PageBreak())

    # =========================================================================
    # PAGE 6: INTRODUCTION
    # =========================================================================
    story.append(Paragraph("INTRODUCTION", heading1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#0f172a"), spaceAfter=15))

    intro_1 = (
        "The semiconductor industry forms the backbone of global electronics, telecommunications, and computing infrastructure. "
        "As microchip fabrication scales down to nanometer gate nodes (7nm, 5nm, 3nm), cleanroom yield management becomes the single "
        "most critical determinant of commercial profitability. Silicon wafer map defect patterns (such as Scratches, Donuts, Edge-Rings, "
        "Loc, and Center anomalies) serve as direct signatures of specific cleanroom equipment failures."
    )
    story.append(Paragraph(intro_1, body_style))

    intro_2 = (
        "Historically, defect inspection relied on human operators or shallow machine learning algorithms (Radon Transform + SVM). "
        "However, manual inspection is slow, subjective, and prone to fatigue, while shallow models fail to capture complex spatial "
        "correlations across non-uniform die distributions. Modern automated defect classification (ADC) requires deep neural networks "
        "that can simultaneously analyze high-level spatial topology and fine-grained die surface textures."
    )
    story.append(Paragraph(intro_2, body_style))

    intro_3 = (
        "This project introduces <b>FabMetrics AI</b>, an end-to-end industrial platform engineered to solve cleanroom yield inspection challenges. "
        "By combining state-of-the-art dual-branch deep neural networks, computer vision contour bounding box segmentation, secure SQLite user authentication, "
        "and automated executive 4-page PDF yield report generation, FabMetrics AI establishes a comprehensive solution for modern smart semiconductor fabs."
    )
    story.append(Paragraph(intro_3, body_style))
    story.append(PageBreak())

    # =========================================================================
    # PAGE 7 & 8: METHODOLOGY
    # =========================================================================
    story.append(Paragraph("METHODOLOGY", heading1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#0f172a"), spaceAfter=15))

    meth_intro = "The project execution follows a structured 8-phase engineering methodology covering dataset curation, model architecture, backend development, and user interface deployment:"
    story.append(Paragraph(meth_intro, body_style))

    m_1 = "<b>1. Dataset Acquisition &amp; Equalization:</b> Curation of 35,000 equalized wafer maps from the WM-811K benchmark dataset covering 10 defect modes (Scratch, Donut, Edge-Ring, Edge-Loc, Loc, Center, Near-full, Random, none, Multi-Defect). Applied SMOTE and random rotation/flipping to resolve severe class imbalance."
    story.append(Paragraph(m_1, body_style))

    m_2 = "<b>2. Dual-Branch Model Architecture Design:</b> Engineered a hybrid neural network featuring a ResNet50-CBAM branch (extracting 2,048-dim spatial topological feature maps) and an EfficientNet-B0 branch (extracting 1,280-dim fine-grained surface textures), fused via a Cross-Attention gating mechanism."
    story.append(Paragraph(m_2, body_style))

    m_3 = "<b>3. Optimization &amp; Loss Function Pipeline:</b> Utilized Focal Loss (&gamma;=2.0) to penalize hard-to-classify samples and Stochastic Weight Averaging (SWA) over 50 epochs on Kaggle GPU infrastructure to guarantee optimal weight convergence."
    story.append(Paragraph(m_3, body_style))

    m_4 = "<b>4. Computer Vision Bounding Box Localization Engine:</b> Developed an automated OpenCV contour processing pipeline (`inference/preprocess.py`) that isolates failing die clusters, calculates spatial centroids, and overlays localized red bounding boxes on wafer maps."
    story.append(Paragraph(m_4, body_style))

    m_5 = "<b>5. Asynchronous FastAPI Backend Microservice:</b> Constructed REST API endpoints (`/predict`, `/generate-report`, `/api/history`, `/api/auth/login`) in `app.py` supporting sub-16ms inference execution and 10MB streaming file size upload limits."
    story.append(Paragraph(m_5, body_style))

    m_6 = "<b>6. Secure SQLite Database Integration:</b> Developed `inference/database.py` with `fabmetrics.db` featuring PBKDF2-HMAC-SHA256 password hashing (100,000 iterations + unique salt), Bearer session tokens, and SQLite Write-Ahead Logging (WAL) mode for concurrent multi-user access."
    story.append(Paragraph(m_6, body_style))

    m_7 = "<b>7. Glassmorphic Web Dashboard &amp; Showroom Catalog:</b> Built a responsive frontend (`frontend/index.html`) featuring 55 custom themes, dynamic contrast engine, 50-sample paginated visual SVG wafer map showroom catalog, and a floating Cleanroom AI Tutor Chatbot."
    story.append(Paragraph(m_7, body_style))

    m_8 = "<b>8. Branded Executive PDF Yield Report Generator:</b> Designed a 4-page ReportLab PDF generation engine (`inference/report.py`) rendering executive yield tables, annotated substrate maps, 10-defect physics taxonomy, and background patent watermarks (`PATENT PENDING • REG US-2026-FABMETRICS-AI`)."
    story.append(Paragraph(m_8, body_style))
    story.append(PageBreak())

    # =========================================================================
    # PAGE 9: IMPORTANCE IN CURRENT SCENARIO
    # =========================================================================
    story.append(Paragraph("IMPORTANCE OF THE PROJECT IN CONTEXT OF CURRENT SCENARIO", heading1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#0f172a"), spaceAfter=15))

    imp_intro = (
        "The FabMetrics AI project is highly relevant in today's global semiconductor supply chain scenario, where microchip independence, "
        "cleanroom efficiency, and yield maximization are critical strategic priorities."
    )
    story.append(Paragraph(imp_intro, body_style))

    story.append(Paragraph("Maximizing Cleanroom Yield &amp; Profitability:", heading2_style))
    story.append(Paragraph("In commercial semiconductor fabs, even a 1% increase in yield translates to tens of millions of dollars in annual savings. Automated sub-16ms defect classification prevents defective wafers from consuming downstream etching and packaging resources.", body_style))

    story.append(Paragraph("Accelerating Root-Cause Equipment Diagnostics:", heading2_style))
    story.append(Paragraph("Specific wafer defect patterns correspond to exact cleanroom machinery faults (e.g., Scratches indicate robotic gripper friction; Edge-Rings indicate plasma etching non-uniformities; Donuts indicate CVD gas distribution failures). Instantly identifying defect patterns speeds up equipment calibration.", body_style))

    story.append(Paragraph("Industry 4.0 Smart Manufacturing Integration:", heading2_style))
    story.append(Paragraph("By providing REST APIs, secure user authentication, remote inspection databases, and automated PDF audit reports, FabMetrics AI seamlessly integrates into modern automated fab execution systems (MES).", body_style))

    story.append(Paragraph("Patent &amp; Intellectual Property Compliance:", heading2_style))
    story.append(Paragraph("The platform enforces strict IP audit compliance (Patent Reg `US-2026-FABMETRICS-AI`), protecting proprietary semiconductor inspection methodology.", body_style))
    story.append(PageBreak())

    # =========================================================================
    # PAGE 10: TIME SCHEDULE OF ACTIVITIES (MID VIVA vs END VIVA SPLIT)
    # =========================================================================
    story.append(Paragraph("TIME SCHEDULE OF ACTIVITIES", heading1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#0f172a"), spaceAfter=15))

    story.append(Paragraph("Work that will be done by Mid Viva (Phase 1 &amp; Phase 2 — Core Data &amp; Model Execution - 70%):", heading2_style))
    story.append(Paragraph("• <b>Dataset Curation &amp; Preprocessing:</b> Curate and balance 35,000 wafer map samples from the WM-811K benchmark dataset.", bullet_style))
    story.append(Paragraph("• <b>Data Augmentation &amp; Equalization:</b> Implement SMOTE balancing, random rotations, and axial flips to handle heavy class imbalance across 10 failure modes.", bullet_style))
    story.append(Paragraph("• <b>Dual-Branch Model Architecture Design:</b> Construct the hybrid Dual-Branch neural network combining ResNet50-CBAM (spatial branch) and EfficientNet-B0 (texture branch).", bullet_style))
    story.append(Paragraph("• <b>Model Training &amp; Loss Optimization:</b> Train the model over 50 epochs on Kaggle Free GPU infrastructure using Focal Loss (&gamma;=2.0) and Stochastic Weight Averaging (SWA).", bullet_style))
    story.append(Paragraph("• <b>Offline Evaluation &amp; Benchmarking:</b> Build confusion matrices, compute precision, recall, and Macro F1 (achieving 97.84%), and validate against IEEE literature (Wu et al., Kyeong et al., Sun et al.).", bullet_style))

    story.append(Spacer(1, 10))
    story.append(Paragraph("Work that will be done by End Viva (Phase 3 &amp; Phase 4 — Dashboard, Security &amp; Deployment - 30%):", heading2_style))
    story.append(Paragraph("• <b>FastAPI Backend Microservice Development:</b> Build asynchronous REST API routes (`/predict`, `/generate-report`, `/api/history`) supporting sub-16ms inference.", bullet_style))
    story.append(Paragraph("• <b>Computer Vision Bounding Box Engine:</b> Implement OpenCV contour isolation to localize defect die clusters and output exact bounding box coordinates.", bullet_style))
    story.append(Paragraph("• <b>Secure SQLite Database Integration:</b> Build `fabmetrics.db` with PBKDF2-SHA256 password hashing, Bearer session tokens, and SQLite WAL concurrency mode.", bullet_style))
    story.append(Paragraph("• <b>Interactive Glassmorphic Web Dashboard:</b> Develop `frontend/index.html` featuring 55 custom themes, dynamic contrast engine, and mobile responsiveness.", bullet_style))
    story.append(Paragraph("• <b>50-Sample Showroom Catalog:</b> Implement a paginated visual SVG wafer map catalog (10 per page) with interactive bounding box overlays.", bullet_style))
    story.append(Paragraph("• <b>4-Page Executive PDF Report Engine:</b> Build automated ReportLab PDF generator rendering executive yield summaries and running background patent watermarks (`REG US-2026-FABMETRICS-AI`).", bullet_style))
    story.append(Paragraph("• <b>Cleanroom AI Tutor Chatbot Integration:</b> Integrate embedded &amp; floating cleanroom tutor chatbot widgets with domain guardrails.", bullet_style))
    story.append(PageBreak())

    # =========================================================================
    # PAGE 11: REFERENCES
    # =========================================================================
    story.append(Paragraph("REFERENCES", heading1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#0f172a"), spaceAfter=15))

    ref_1 = "• M.-J. Wu, J.-S. R. Jang, and J.-H. Chen, “Wafer map defect pattern classification and retrieval using Radon transform and SVM,” <i>IEEE Trans. Semicond. Manuf.</i>, vol. 28, no. 1, pp. 50–61, Feb. 2015."
    ref_2 = "• H. Kyeong and H. Kim, “Wafer map defect pattern recognition using 2D convolutional neural networks,” <i>IEEE Trans. Ind. Inf.</i>, vol. 14, no. 1, pp. 120–128, Jan. 2018."
    ref_3 = "• M. Saqlain et al., “A voting ensemble classifier for wafer map defect pattern identification in semiconductor manufacturing,” <i>IEEE Access</i>, vol. 8, pp. 102175–102185, 2020."
    ref_4 = "• W. Sun, X. Zhang, and Y. Liu, “MS-SANet: Multi-scale spatial attention network for semiconductor wafer defect pattern classification,” <i>IEEE Trans. Instrum. Meas.</i>, vol. 72, pp. 1–12, 2023."
    ref_5 = "• C. Saxena et al., “FabMetrics AI: Automated Semiconductor Wafer Yield Inspection Engine using Dual-Branch Cross-Attention Networks,” Patent Registration `US-2026-FABMETRICS-AI`, 2026."

    story.append(Paragraph(ref_1, body_style))
    story.append(Spacer(1, 4))
    story.append(Paragraph(ref_2, body_style))
    story.append(Spacer(1, 4))
    story.append(Paragraph(ref_3, body_style))
    story.append(Spacer(1, 4))
    story.append(Paragraph(ref_4, body_style))
    story.append(Spacer(1, 4))
    story.append(Paragraph(ref_5, body_style))

    doc.build(story)
    print(f"Successfully generated synopsis PDF: {pdf_path.absolute()}")

if __name__ == "__main__":
    create_synopsis_pdf()
