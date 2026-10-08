#!/usr/bin/env python3
"""
COLONVISION AI — Professional Clinical Analysis Report Generator
================================================================
Generates high-fidelity, research-oriented PDF analysis reports
using ReportLab for verified colonoscopy image evaluations.

Architecture: ResNet-50 + Grad-CAM Explainability + Endoscopy Domain Verification
"""

import io
from datetime import datetime
from pathlib import Path
from typing import Dict, Union, Optional

from PIL import Image as PILImage
import numpy as np

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image as RLImage,
    KeepTogether,
    HRFlowable,
)
from reportlab.pdfgen import canvas

# ── Color Palette Constants ───────────────────────────────────────────────────
COLOR_PRIMARY = colors.HexColor("#174A3A")       # Deep Forest Green
COLOR_PRIMARY_DARK = colors.HexColor("#10382C")  # Dark Forest Green
COLOR_PRIMARY_LIGHT = colors.HexColor("#E3EEE8") # Soft Mint Accent
COLOR_TEXT = colors.HexColor("#151A17")          # Dark Charcoal Text
COLOR_MUTED = colors.HexColor("#69716C")         # Slate Muted Text
COLOR_BORDER = colors.HexColor("#E1E5E1")        # Subtle Gray Border
COLOR_BG_LIGHT = colors.HexColor("#F7F7F5")      # Light Off-White
COLOR_SUCCESS = colors.HexColor("#26704F")       # Verified Green
COLOR_WARNING = colors.HexColor("#9A6B32")       # Amber Warning
COLOR_ERROR = colors.HexColor("#A34848")         # Rejection Crimson


class NumberedCanvas(canvas.Canvas):
    """Two-pass canvas for dynamic 'Page X of Y' numbering and clean running headers."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count: int):
        self.saveState()
        self.setFont("Helvetica-Bold", 7)
        self.setFillColor(COLOR_PRIMARY)
        self.drawString(54, 755, "COLONVISION AI")
        self.setFont("Helvetica", 7)
        self.setFillColor(COLOR_MUTED)
        self.drawString(135, 755, "|   EXPLAINABLE COLONOSCOPY IMAGE ANALYSIS REPORT")
        self.drawRightString(558, 755, "RESEARCH PROTOTYPE")

        self.setStrokeColor(COLOR_BORDER)
        self.setLineWidth(0.6)
        self.line(54, 747, 558, 747)

        # Bottom footer
        self.line(54, 45, 558, 45)
        self.setFont("Helvetica", 7)
        self.setFillColor(COLOR_MUTED)
        self.drawString(54, 33, "Research Prototype — Not for Clinical Diagnosis or Treatment Decisions")
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(558, 33, page_str)
        self.restoreState()


def _pil_to_reportlab_image(pil_img: PILImage.Image, width_pt: float, height_pt: float) -> RLImage:
    """Helper to convert in-memory PIL image into ReportLab Flowable Image."""
    img_byte_arr = io.BytesIO()
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    pil_img.save(img_byte_arr, format="PNG")
    img_byte_arr.seek(0)
    return RLImage(img_byte_arr, width=width_pt, height=height_pt)


def generate_analysis_report(
    original_image: PILImage.Image,
    gradcam_image: PILImage.Image,
    overlay_image: PILImage.Image,
    prediction: str,
    probabilities: Union[Dict[str, float], np.ndarray, list],
    confidence: float,
    validation_score: float,
    model_name: str = "ResNet-50",
    analysis_timestamp: Optional[datetime] = None,
    threshold: float = 0.6800,
    input_resolution: str = "224 × 224",
    target_layer: str = "model.layer4[-1]",
) -> bytes:
    """
    Generates a professional multi-section clinical analysis report in PDF format.
    
    Returns:
        bytes: Raw PDF file bytes ready for download or disk persistence.
    """
    if analysis_timestamp is None:
        analysis_timestamp = datetime.now()

    date_str = analysis_timestamp.strftime("%d %B %Y")
    time_str = analysis_timestamp.strftime("%H:%M")

    # Format probabilities
    if isinstance(probabilities, dict):
        prob_divert = float(probabilities.get("colon_diverticula", 0.0))
        prob_cancer = float(probabilities.get("colorectal_cancer", 0.0))
    elif hasattr(probabilities, "__len__"):
        prob_divert = float(probabilities[0])
        prob_cancer = float(probabilities[1])
    else:
        prob_divert, prob_cancer = 0.0, 0.0

    # Ensure formatted prediction name
    norm_pred = prediction.strip().title()
    if "Cancer" in norm_pred:
        formatted_pred = "Colorectal Cancer"
        alt_class = "Colon Diverticula"
        alt_prob = prob_divert
    else:
        formatted_pred = "Colon Diverticula"
        alt_class = "Colorectal Cancer"
        alt_prob = prob_cancer

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54,
    )

    styles = getSampleStyleSheet()

    # Custom Typography Styles
    style_title = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=COLOR_TEXT,
    )
    style_subtitle = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=14,
        textColor=COLOR_PRIMARY,
    )
    style_meta_head = ParagraphStyle(
        "DocMetaHead",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=11,
        textColor=COLOR_MUTED,
    )
    style_sec_header = ParagraphStyle(
        "DocSecHeader",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=10,
        leading=13,
        textColor=COLOR_PRIMARY_DARK,
        spaceBefore=8,
        spaceAfter=4,
    )
    style_body = ParagraphStyle(
        "DocBody",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=12,
        textColor=COLOR_TEXT,
    )
    style_disclaimer = ParagraphStyle(
        "DocDisclaimer",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7.2,
        leading=10,
        textColor=COLOR_MUTED,
    )

    story = []

    # ── Header Block ──────────────────────────────────────────────────────────
    story.append(Paragraph("COLONVISION AI", style_subtitle))
    story.append(Paragraph("Explainable Colonoscopy Image Analysis Report", style_title))
    story.append(Spacer(1, 4))
    story.append(Paragraph(f"Research Analysis Report &nbsp;&bull;&nbsp; Date: <b>{date_str}</b> &nbsp;&bull;&nbsp; Time: <b>{time_str}</b>", style_meta_head))
    story.append(Spacer(1, 8))
    story.append(HRFlowable(width="100%", thickness=1, color=COLOR_BORDER, spaceBefore=2, spaceAfter=8))

    # ── Status & Analysis Summary Strip ───────────────────────────────────────
    status_summary_table = Table(
        [
            [
                Paragraph("<b>ANALYSIS STATUS</b>", style_sec_header),
                Paragraph("<b>ANALYSIS SUMMARY</b>", style_sec_header),
            ],
            [
                Paragraph(
                    "<font color='#26704F'>&#10003;</font> Endoscopy domain verified<br/>"
                    "<font color='#26704F'>&#10003;</font> Model analysis completed<br/>"
                    "<font color='#26704F'>&#10003;</font> Explainability generated",
                    style_body,
                ),
                Paragraph(
                    f"&bull; <b>Prediction:</b> <b>{formatted_pred}</b><br/>"
                    f"&bull; <b>Confidence:</b> <b>{confidence * 100:.2f}%</b><br/>"
                    f"&bull; <b>Alternative class:</b> {alt_class}<br/>"
                    f"&bull; <b>Probability:</b> {alt_prob * 100:.2f}%<br/>"
                    f"&bull; <b>Model:</b> {model_name}<br/>"
                    f"&bull; <b>Input:</b> {input_resolution}",
                    style_body,
                ),
            ],
        ],
        colWidths=[240, 264],
    )
    status_summary_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), COLOR_BG_LIGHT),
            ("BOX", (0, 0), (-1, -1), 0.5, COLOR_BORDER),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ])
    )
    story.append(status_summary_table)
    story.append(Spacer(1, 10))

    # ── Section 01: Domain Verification ───────────────────────────────────────
    story.append(Paragraph("<b>01 &mdash; DOMAIN VERIFICATION</b>", style_sec_header))
    domain_table_data = [
        [
            Paragraph("<b>Verification Status:</b>", style_body),
            Paragraph("<font color='#26704F'><b>VERIFIED (ENDOSCOPY DOMAIN)</b></font>", style_body),
            Paragraph("<b>Validation Score:</b>", style_body),
            Paragraph(f"<b>{validation_score:.4f}</b> (Threshold &ge; {threshold:.4f})", style_body),
        ]
    ]
    domain_table = Table(domain_table_data, colWidths=[120, 160, 100, 124])
    domain_table.setStyle(
        TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
            ("TOPPADDING", (0, 0), (-1, -1), 2),
        ])
    )
    story.append(domain_table)
    story.append(Spacer(1, 3))
    story.append(
        Paragraph(
            "The submitted image passed the endoscopy-domain verification stage and was therefore processed by the classification model. "
            "Domain validation confirms structural and color characteristics matching endoscopy reference distributions.",
            style_meta_head,
        )
    )
    story.append(Spacer(1, 8))

    # ── Section 02: Model Prediction & Probability Table ───────────────────────
    story.append(Paragraph("<b>02 &mdash; MODEL PREDICTION</b>", style_sec_header))

    prob_table_data = [
        [Paragraph("<b>Target Classification</b>", style_body), Paragraph("<b>Differential Probability</b>", style_body), Paragraph("<b>Model Confidence</b>", style_body)],
        [
            Paragraph("<b>Colorectal Cancer</b> (Malignant Pathology)", style_body),
            Paragraph(f"<b>{prob_cancer * 100:.2f}%</b>", style_body),
            Paragraph(f"{'PRIMARY PREDICTION' if formatted_pred == 'Colorectal Cancer' else '&mdash;'}", style_meta_head),
        ],
        [
            Paragraph("<b>Colon Diverticula</b> (Benign Outpouching)", style_body),
            Paragraph(f"<b>{prob_divert * 100:.2f}%</b>", style_body),
            Paragraph(f"{'PRIMARY PREDICTION' if formatted_pred == 'Colon Diverticula' else '&mdash;'}", style_meta_head),
        ],
    ]
    prob_table = Table(prob_table_data, colWidths=[240, 140, 124])
    prob_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), COLOR_PRIMARY_LIGHT),
            ("BOX", (0, 0), (-1, -1), 0.5, COLOR_BORDER),
            ("GRID", (0, 0), (-1, -1), 0.5, COLOR_BORDER),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ])
    )
    story.append(prob_table)
    story.append(Spacer(1, 10))

    # ── Section 03: Visual Explanation (Grad-CAM Artifacts) ───────────────────
    story.append(Paragraph("<b>03 &mdash; MODEL EXPLAINABILITY (GRAD-CAM)</b>", style_sec_header))
    story.append(
        Paragraph(
            f"Explainability Method: <b>Grad-CAM</b> &nbsp;|&nbsp; Target Convolutional Layer: <code>{target_layer}</code> &nbsp;|&nbsp; Feature Resolution: <b>7 &times; 7</b><br/>"
            "Grad-CAM provides a visual indication of image regions that contributed most strongly to the model prediction. "
            "Highlighted model-attended regions represent gradient activation patterns and are not intended as diagnostic lesion boundary segmentations.",
            style_meta_head,
        )
    )
    story.append(Spacer(1, 6))

    # Images formatted side-by-side: 3 images fitting into 504 pt width (160 pt each)
    img_size_pt = 158.0
    try:
        rl_orig = _pil_to_reportlab_image(original_image, img_size_pt, img_size_pt)
    except Exception:
        rl_orig = Paragraph("Visualization unavailable.", style_meta_head)

    try:
        rl_cam = _pil_to_reportlab_image(gradcam_image, img_size_pt, img_size_pt)
    except Exception:
        rl_cam = Paragraph("Visualization unavailable.", style_meta_head)

    try:
        rl_overlay = _pil_to_reportlab_image(overlay_image, img_size_pt, img_size_pt)
    except Exception:
        rl_overlay = Paragraph("Visualization unavailable.", style_meta_head)

    images_table = Table(
        [
            [rl_orig, rl_cam, rl_overlay],
            [
                Paragraph("<b>01 &middot; Preprocessed Input Crop</b><br/><font color='#69716C'>Standardized 224&times;224</font>", style_meta_head),
                Paragraph("<b>02 &middot; Grad-CAM Heatmap</b><br/><font color='#69716C'>Jet Colormap Activation</font>", style_meta_head),
                Paragraph("<b>03 &middot; Composite Attention Overlay</b><br/><font color='#69716C'>Alpha Blended Visualization</font>", style_meta_head),
            ],
        ],
        colWidths=[168, 168, 168],
    )
    images_table.setStyle(
        TableStyle([
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 4),
            ("TOPPADDING", (0, 1), (-1, 1), 2),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ])
    )
    story.append(images_table)
    story.append(Spacer(1, 10))

    # ── Section 04: Analysis Details ──────────────────────────────────────────
    story.append(Paragraph("<b>04 &mdash; ANALYSIS DETAILS</b>", style_sec_header))

    metadata_table_data = [
        [
            Paragraph("<b>Analysis date/time:</b>", style_meta_head),
            Paragraph(f"{date_str} &middot; {time_str}", style_body),
            Paragraph("<b>Model:</b>", style_meta_head),
            Paragraph(model_name, style_body),
        ],
        [
            Paragraph("<b>Input resolution:</b>", style_meta_head),
            Paragraph(input_resolution, style_body),
            Paragraph("<b>Domain verification status:</b>", style_meta_head),
            Paragraph(f"VERIFIED (Score: {validation_score:.4f})", style_body),
        ],
        [
            Paragraph("<b>Prediction:</b>", style_meta_head),
            Paragraph(f"<b>{formatted_pred}</b>", style_body),
            Paragraph("<b>Confidence:</b>", style_meta_head),
            Paragraph(f"<b>{confidence * 100:.2f}%</b>", style_body),
        ],
        [
            Paragraph("<b>Explainability status:</b>", style_meta_head),
            Paragraph("Grad-CAM Available", style_body),
            Paragraph("<b>Explainability Target:</b>", style_meta_head),
            Paragraph(target_layer, style_body),
        ],
    ]
    meta_table = Table(metadata_table_data, colWidths=[120, 140, 130, 114])
    meta_table.setStyle(
        TableStyle([
            ("BOX", (0, 0), (-1, -1), 0.5, COLOR_BORDER),
            ("BACKGROUND", (0, 0), (-1, -1), COLOR_BG_LIGHT),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ])
    )
    story.append(meta_table)
    story.append(Spacer(1, 10))

    # ── Disclaimer & Privacy Notice ───────────────────────────────────────────
    disclaimer_text = (
        "<b>RESEARCH DISCLAIMER & PRIVACY NOTICE:</b><br/>"
        "Research prototype. This system is intended for research and educational purposes only and is not intended for "
        "clinical diagnosis, triage, treatment decisions, or replacement of professional medical judgment. "
        "The model prediction represents the output of an experimental machine-learning system and should not be interpreted "
        "as a medical diagnosis. Images are processed within the application workflow and are not intentionally stored as part of this report-generation process."
    )
    story.append(
        Table(
            [[Paragraph(disclaimer_text, style_disclaimer)]],
            colWidths=[504],
            style=[
                ("BOX", (0, 0), (-1, -1), 0.5, COLOR_BORDER),
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#FFFFFF")),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ],
        )
    )

    doc.build(story, canvasmaker=NumberedCanvas)
    buffer.seek(0)
    return buffer.getvalue()


if __name__ == "__main__":
    # Self-test routine
    dummy_img = PILImage.new("RGB", (224, 224), color=(120, 40, 40))
    pdf_bytes = generate_analysis_report(
        original_image=dummy_img,
        gradcam_image=dummy_img,
        overlay_image=dummy_img,
        prediction="Colorectal Cancer",
        probabilities={"colon_diverticula": 0.0997, "colorectal_cancer": 0.9003},
        confidence=0.9003,
        validation_score=0.8111,
        model_name="ResNet-50",
    )
    print(f"Self-test: Generated PDF with {len(pdf_bytes)} bytes.")
