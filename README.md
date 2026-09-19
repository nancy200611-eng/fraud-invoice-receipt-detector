---
title: Invoice Fraud Detector
emoji: 🛡️
colorFrom: blue
colorTo: indigo
sdk: gradio
app_file: app.py
pinned: false
---

# 🛡️ End-to-End Fraud Invoice & Receipt Detection Platform

An enterprise-grade document forensics and tampering detection system combining **Deep Learning (ResNet-18 + Grad-CAM)**, **Digital Image Forensics (Error Level Analysis - ELA & SRM high-pass filter residuals)**, **Optical Character Recognition (EasyOCR)**, and **Heuristic Business Rule Auditing**.

---

## 🚀 System Architecture

```
E:\fraud Invoice&Receipt Detector/
├── data/
│   ├── raw/                  # Genuine authentic receipts/invoices
│   ├── ocr_extracted/        # CSV token database (bounding boxes, confidences)
│   ├── manipulated/
│   │   ├── images/           # Paired tampered documents (forged totals, inpainted fields)
│   │   └── masks/            # Ground-truth binary manipulation masks
│   ├── samples/              # Quick test visual artifacts
│   └── audit_ledger.db       # SQLite ledger for duplicate & tamper detection
├── src/
│   ├── ocr/                  # Phase 1: EasyOCR pipeline & synthetic generator
│   ├── manipulation/         # Phase 2: Inpainting, text replacement, JPEG mismatch
│   ├── forensics/            # Phase 3 & 5: ELA, SRM filter bank, noise analysis, resilience
│   ├── models/               # Phase 3: ResNet-18 classifier, Grad-CAM, U-Net localizer
│   ├── rules/                # Phase 4: Business math validation & duplicate detection
│   └── backend/              # Phase 4: FastAPI REST API & interactive dashboard
├── models/
│   └── resnet18_fraud.pth    # Fine-tuned PyTorch classifier weights
├── tests/
│   └── test_system.py        # Automated test suite
├── run_pipeline.py           # Unified CLI orchestration tool
└── requirements.txt
```

---

## ⚡ Quick Start

### 1. Launch the Interactive Web Dashboard
```bash
python run_pipeline.py --serve
```
Open your browser at **`http://127.0.0.1:8000`**.

### 2. Run Automated Test Suite
```bash
python run_pipeline.py --test
```

### 3. Run Individual Pipeline Phases
- **Phase 1 (OCR & Dataset Lift-Off)**:
  ```bash
  python run_pipeline.py --phase 1
  ```
- **Phase 2 (Manipulation & Gravity-Defying Fakes)**:
  ```bash
  python run_pipeline.py --phase 2
  ```
- **Phase 3 (Forensics, ResNet-18 Training & Grad-CAM)**:
  ```bash
  python run_pipeline.py --phase 3
  ```
- **Phase 4 & 5 (Business Rules & Provenance Shield)**:
  ```bash
  python run_pipeline.py --phase 4
  python run_pipeline.py --phase 5
  ```

---

## 🔍 Core Detection Capabilities

1. **Digital Image Forensics (ELA & SRM Residuals)**:
   - **Error Level Analysis (ELA)**: Recompresses images at 90% JPEG quality to highlight compression rate differentials between original invoice paper and newly inserted numbers.
   - **Spatial Rich Models (SRM)**: 1st and 2nd-order high-pass kernels highlight edge discontinuities and splicing boundaries.
2. **ResNet-18 Classifier & Grad-CAM Explainer**:
   - Classifies documents as Genuine vs Manipulated.
   - Grad-CAM hooks into the final convolutional layer (`layer4`) to visually highlight the exact pixels that triggered the fraud verdict.
3. **Dual-Stream Encoder-Decoder (Localizer)**:
   - Fuses RGB pixels with forensic residual channels to produce pixel-level localization masks.
4. **Heuristic & Ledger Integrity**:
   - Math audit: Verifies $\text{Subtotal} + \text{Tax} == \text{Total Due}$.
   - Duplicate detection: Checks invoice number and 64-bit perceptual hash (`pHash`) against an SQLite database.
5. **Anti-Gravity Provenance Shield**:
   - Generates and verifies HMAC-SHA256 digital seals for tamper-evident tracking.
