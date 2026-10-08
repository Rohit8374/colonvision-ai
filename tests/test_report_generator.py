#!/usr/bin/env python3
"""
Unit and Integration Tests for Professional PDF Report Generator
=================================================================
Validates:
- Successful PDF generation
- PDF is non-empty and well-formed
- PDF can be opened and parsed by a standard PDF reader
- Presence of mandatory sections and headers
- Correct prediction rendering (Colorectal Cancer vs Colon Diverticula)
- Dynamic confidence and probability values (no hardcoding)
- Domain verification details and threshold
- Model explainability metadata
- Mandatory research disclaimer and privacy statements
- Graceful handling of missing/corrupt visualization images
"""

import io
import sys
from datetime import datetime
from pathlib import Path
import unittest

from PIL import Image
from pypdf import PdfReader

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.report_generator import generate_analysis_report


class TestReportGenerator(unittest.TestCase):
    def setUp(self):
        # Create valid synthetic test images
        self.orig_img = Image.new("RGB", (224, 224), color=(180, 80, 70))
        self.cam_img = Image.new("RGB", (224, 224), color=(0, 200, 100))
        self.overlay_img = Image.new("RGB", (224, 224), color=(120, 140, 90))

    def test_pdf_generation_cancer_success(self):
        """Test PDF generation for a Colorectal Cancer inference result."""
        pdf_bytes = generate_analysis_report(
            original_image=self.orig_img,
            gradcam_image=self.cam_img,
            overlay_image=self.overlay_img,
            prediction="Colorectal Cancer",
            probabilities={"colon_diverticula": 0.0997, "colorectal_cancer": 0.9003},
            confidence=0.9003,
            validation_score=0.8115,
            model_name="ResNet-50",
            analysis_timestamp=datetime(2026, 10, 8, 14, 30),
            threshold=0.6800,
        )

        # 1. Non-empty check
        self.assertIsInstance(pdf_bytes, bytes)
        self.assertGreater(len(pdf_bytes), 3000, "PDF byte stream is smaller than expected")

        # 2. PDF can be opened and read
        reader = PdfReader(io.BytesIO(pdf_bytes))
        self.assertGreaterEqual(len(reader.pages), 1, "PDF must have at least 1 page")

        # Extract full document text and normalize whitespace
        import re
        raw_text = " ".join([page.extract_text() for page in reader.pages])
        full_text = re.sub(r"\s+", " ", raw_text)

        # 3. Expected header and structure
        self.assertIn("COLONVISION AI", full_text)
        self.assertIn("Explainable Colonoscopy Image Analysis Report", full_text)
        self.assertIn("Research Analysis Report", full_text)
        self.assertIn("ANALYSIS STATUS", full_text)
        self.assertIn("Endoscopy domain verified", full_text)
        self.assertIn("Model analysis completed", full_text)

        # 4. Correct prediction & confidence
        self.assertIn("Colorectal Cancer", full_text)
        self.assertIn("90.03%", full_text)
        self.assertIn("Colon Diverticula", full_text)
        self.assertIn("9.97%", full_text)

        # 5. Domain verification section
        self.assertIn("DOMAIN VERIFICATION", full_text)
        self.assertIn("0.8115", full_text)
        self.assertIn("0.6800", full_text)
        self.assertIn("The submitted image passed the endoscopy-domain verification stage", full_text)

        # 6. Model explainability section
        self.assertIn("MODEL EXPLAINABILITY", full_text)
        self.assertIn("Grad-CAM", full_text)
        self.assertIn("layer4[-1]", full_text)
        self.assertIn("model-attended regions", full_text)

        # 7. Analysis details section
        self.assertIn("ANALYSIS DETAILS", full_text)
        self.assertIn("ResNet-50", full_text)
        self.assertIn("224 × 224", full_text)

        # 8. Research disclaimer & privacy
        self.assertIn("Research prototype", full_text)
        self.assertIn("not intended for clinical diagnosis", full_text)
        self.assertIn("Images are processed within the application workflow", full_text)

    def test_pdf_generation_diverticula_success(self):
        """Test PDF generation for a Colon Diverticula inference result."""
        pdf_bytes = generate_analysis_report(
            original_image=self.orig_img,
            gradcam_image=self.cam_img,
            overlay_image=self.overlay_img,
            prediction="Colon Diverticula",
            probabilities={"colon_diverticula": 0.8850, "colorectal_cancer": 0.1150},
            confidence=0.8850,
            validation_score=0.7950,
            model_name="ResNet-50",
            analysis_timestamp=datetime(2026, 10, 8, 16, 45),
            threshold=0.6800,
        )

        self.assertIsInstance(pdf_bytes, bytes)
        self.assertGreater(len(pdf_bytes), 3000)

        reader = PdfReader(io.BytesIO(pdf_bytes))
        full_text = " ".join([page.extract_text() for page in reader.pages])

        self.assertIn("Colon Diverticula", full_text)
        self.assertIn("88.50%", full_text)
        self.assertIn("0.7950", full_text)

    def test_no_debug_or_developer_leaks(self):
        """Verify report does not leak internal developer/system debugging parameters."""
        pdf_bytes = generate_analysis_report(
            original_image=self.orig_img,
            gradcam_image=self.cam_img,
            overlay_image=self.overlay_img,
            prediction="Colorectal Cancer",
            probabilities={"colon_diverticula": 0.05, "colorectal_cancer": 0.95},
            confidence=0.95,
            validation_score=0.85,
        )
        reader = PdfReader(io.BytesIO(pdf_bytes))
        full_text = " ".join([page.extract_text() for page in reader.pages])

        # Developer debug terms that must NOT appear
        self.assertNotIn("learning_rate", full_text.lower())
        self.assertNotIn("batch_size", full_text.lower())
        self.assertNotIn("optimizer", full_text.lower())
        self.assertNotIn("cuda:0", full_text.lower())
        self.assertNotIn("gpu memory", full_text.lower())

    def test_graceful_missing_image_fallback(self):
        """Verify generator handles None/unusable image without unhandled crash."""
        pdf_bytes = generate_analysis_report(
            original_image=None,  # Intentionally None
            gradcam_image=self.cam_img,
            overlay_image=self.overlay_img,
            prediction="Colorectal Cancer",
            probabilities={"colon_diverticula": 0.10, "colorectal_cancer": 0.90},
            confidence=0.90,
            validation_score=0.82,
        )
        self.assertIsInstance(pdf_bytes, bytes)
        self.assertGreater(len(pdf_bytes), 1000)

        reader = PdfReader(io.BytesIO(pdf_bytes))
        full_text = " ".join([page.extract_text() for page in reader.pages])
        self.assertIn("Visualization unavailable", full_text)


if __name__ == "__main__":
    unittest.main()
