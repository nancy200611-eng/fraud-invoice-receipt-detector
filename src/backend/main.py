import os
import io
import base64
import shutil
import glob
import cv2
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

from src.ocr.ocr_pipeline import ReceiptOCRPipeline
from src.forensics.image_forensics import DocumentForensics
from src.forensics.ai_generator_detector import AIGeneratorDetector
from src.forensics.c2pa_exif_fast_guard import C2PAExifFastGuard
from src.models.fusion_plus import FusionPlusInference
from src.models.classifier import FraudModelInference
from src.models.gradcam import GradCAM
from src.rules.heuristic_engine import HeuristicAuditEngine
from src.forensics.robustness import ShieldRobustnessEngine

app = FastAPI(title="Receipt & Invoice Fraud Detector", version="3.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.environ.get(
    "BASE_DIR",
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
UPLOAD_DIR = os.path.join(BASE_DIR, "data", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

SROIE_DIR = os.environ.get("SROIE_DIR", r"D:\SOIRE\ICDAR-2019-SROIE-master\data\img")
REAL_INV_DIR = os.environ.get("REAL_INV_DIR", r"D:\Receipt-Invoice-fraud-detection\data\real\invoice")
FAKE_INV_DIR = os.environ.get("FAKE_INV_DIR", r"D:\Receipt-Invoice-fraud-detection\data\fake\invoice")
FAKE_REC_DIR = os.environ.get("FAKE_REC_DIR", r"D:\Receipt-Invoice-fraud-detection\data\fake\receipt")

print("Initializing Detection Engines...")
c2pa_guard = C2PAExifFastGuard()
ocr_engine = ReceiptOCRPipeline(use_gpu=False)
forensics_engine = DocumentForensics()
ai_detector = AIGeneratorDetector()

fusion_weights = os.path.join(BASE_DIR, "models", "fusion_plus_weights.pth")
fusion_engine = FusionPlusInference(weights_path=fusion_weights if os.path.exists(fusion_weights) else None)

resnet_weights = os.path.join(BASE_DIR, "models", "resnet18_fraud.pth")
classifier = FraudModelInference(weights_path=resnet_weights if os.path.exists(resnet_weights) else None)
gradcam_engine = GradCAM(classifier.model)

heuristic_engine = HeuristicAuditEngine()
shield_engine = ShieldRobustnessEngine()
print("All Engines Online (Fast-Path C2PA/EXIF Guard Active).")

def mat_to_base64(img, ext=".jpg"):
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    _, buffer = cv2.imencode(ext, img)
    return f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"

@app.get("/api/health")
def health():
    return {"status": "ready"}

@app.post("/api/scan")
async def scan_invoice(file: UploadFile = File(...)):
    filename = file.filename or "uploaded_invoice.jpg"
    save_path = os.path.join(UPLOAD_DIR, filename)
    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return process_document(save_path, filename)

@app.get("/api/scan-sample/{sample_type}")
def scan_sample(sample_type: str):
    if sample_type == "genuine":
        path = os.path.join(SROIE_DIR, "000.jpg")
        if not os.path.exists(path):
            path = glob.glob(os.path.join(BASE_DIR, "data", "raw", "*.jpg"))[0]
        filename = os.path.basename(path)
    else:
        path = os.path.join(FAKE_INV_DIR, "fake_invoice_0000.jpg")
        if not os.path.exists(path):
            path = glob.glob(os.path.join(BASE_DIR, "data", "manipulated", "images", "*.jpg"))[0]
        filename = os.path.basename(path)

    return process_document(path, filename)

def process_document(image_path, filename):
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise HTTPException(status_code=400, detail="Unable to read document image.")

    # =========================================================================
    # FAST-PATH ZERO-LATENCY SHORT-CIRCUIT: C2PA & EXIF PROVENANCE
    # If C2PA or EXIF metadata says AI-generated -> don't waste time, flag instantly!
    # =========================================================================
    fast_res = c2pa_guard.evaluate_fast_path(image_path)
    is_fast_ai, fast_title, fast_explanation = fast_res
    diagnostics = getattr(fast_res, "diagnostics", {})

    if is_fast_ai:
        # Create an instant red warning overlay on the document
        alert_overlay = img_bgr.copy()
        h, w = alert_overlay.shape[:2]
        banner_h = max(60, int(h * 0.08))
        cv2.rectangle(alert_overlay, (0, 0), (w, banner_h), (0, 0, 210), -1)
        cv2.putText(alert_overlay, "C2PA / EXIF PROVENANCE ALERT: AI-GENERATED SYNTHETIC MEDIA", (20, int(banner_h * 0.65)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        blended = cv2.addWeighted(alert_overlay, 0.85, img_bgr, 0.15, 0)

        # Record to ledger
        with open(image_path, "rb") as f:
            file_sha = shield_engine.sign_document(f.read()[:512])
        heuristic_engine.record_scan(
            filename=filename,
            inv_num="AI-SYNTHETIC",
            sha256=file_sha,
            phash="0"*64,
            total=0.0,
            score=1.0 # 99% fraud
        )

        return {
            "filename": filename,
            "fraud_percentage": 99.0,
            "verdict": "HIGH_FRAUD_RISK",
            "status_label": "Confirmed AI-Generated (C2PA / EXIF Fast-Path)",
            "badge_color": "red",
            "summary": "Immediate Short-Circuit Rejection: The document contains verified C2PA Content Credentials, truncated JUMBF blocks, or EXIF metadata proving synthetic media creation. Processing halted immediately.",
            "reasons": [{
                "type": "C2PA_PROVENANCE_ALERT",
                "severity": "CRITICAL",
                "title": fast_title,
                "explanation": fast_explanation
            }],
            "parsed_fields": {"invoice_number": "N/A (AI Generated)", "subtotal": None, "tax": None, "total": None},
            "byte_offset_diagnostics": diagnostics.get("byte_offset_diagnostics", []),
            "whitelist_status": diagnostics.get("whitelist_status"),
            "image_original": mat_to_base64(img_bgr),
            "image_highlighted": mat_to_base64(blended)
        }

    # =========================================================================
    # FULL DEEP ANALYSIS (For documents without explicit C2PA/EXIF AI tags)
    # =========================================================================

    # 1. OCR Extraction
    ocr_tokens = ocr_engine.extract_from_image(image_path)

    # 2. Digital Forensics (Bayar, SRM, HOG, ELA)
    forensic_res = forensics_engine.analyze_document(image_path)

    # 3. AI Generator & Diffusion Signature Detection
    ai_res = ai_detector.detect_ai(image_path, ocr_tokens)

    # 4. Neural FUSION++ & ResNet-18
    fusion_res = fusion_engine.predict(image_path)
    pil_img = Image.open(image_path).convert("RGB")
    tensor = classifier.transform(pil_img).unsqueeze(0).to(classifier.device)
    cam_map = gradcam_engine.generate_heatmap(tensor, target_class=1)
    gradcam_overlay, _ = gradcam_engine.overlay_on_image(image_path, cam_map, alpha=0.50)

    # 5. Accounting Business Logic & Math Audit
    audit_res = heuristic_engine.audit_document(image_path, ocr_tokens)
    parsed = audit_res["parsed_fields"]

    # 6. Compute % OF FAKENESS / FRAUD
    neural_fraud_risk = fusion_res["tamper_probability"]
    resnet_fraud_risk = classifier.predict(image_path)["tamper_probability"]
    forensic_anomaly_risk = min(100.0, forensic_res["forensic_score"] * 100.0)
    rule_penalty = audit_res["risk_penalty"]

    raw_fraud_pct = (neural_fraud_risk * 0.25) + (resnet_fraud_risk * 0.25) + (forensic_anomaly_risk * 0.25) + (rule_penalty * 0.25)

    has_math_error = any(f["code"] == "MATH_DISCREPANCY" for f in audit_res["flags"])
    has_duplicate = any(f["code"] in ["DUPLICATE_INVOICE_ID", "DUPLICATE_IMAGE_REUSE"] for f in audit_res["flags"])

    reasons = []

    # Priority 1: AI Generated Receipt
    if ai_res["is_ai_generated"]:
        raw_fraud_pct = max(raw_fraud_pct, 95.0)
        reasons.append({
            "type": "AI_GENERATED",
            "severity": "CRITICAL",
            "title": f"AI-Generated Document ({ai_res['engine_detected']})",
            "explanation": f"This document was generated by an AI model ({ai_res['engine_detected']}). Specific evidence: " + "; ".join(ai_res["reasons"])
        })

    # Priority 2: Math Discrepancy
    if has_math_error:
        raw_fraud_pct = max(raw_fraud_pct, 90.0)
        for f in audit_res["flags"]:
            if f["code"] == "MATH_DISCREPANCY":
                reasons.append({
                    "type": "MATHEMATICAL_MISMATCH",
                    "severity": "CRITICAL",
                    "title": "Calculated Amount Mismatch",
                    "explanation": f["message"]
                })

    # Priority 3: Duplicate Submission
    if has_duplicate:
        raw_fraud_pct = max(raw_fraud_pct, 85.0)
        for f in audit_res["flags"]:
            if f["code"] in ["DUPLICATE_INVOICE_ID", "DUPLICATE_IMAGE_REUSE"]:
                reasons.append({
                    "type": "DUPLICATE_SUBMISSION",
                    "severity": "CRITICAL",
                    "title": "Duplicate Submission",
                    "explanation": f["message"]
                })

    # Priority 4: Forensics
    if forensic_res["forensic_score"] > 0.35 and not ai_res["is_ai_generated"]:
        reasons.append({
            "type": "IMAGE_FORENSICS",
            "severity": "HIGH",
            "title": "Digital Image Tampering Detected",
            "explanation": f"Discrepancies detected in paper noise patterns and JPEG compression (Bayar anomaly: {forensic_res['bayar_metrics']['bayar_anomaly']}, ELA ratio: {forensic_res['ela_metrics']['high_error_ratio']})."
        })

    # Priority 5: Deep Learning Anomaly
    if (neural_fraud_risk > 50.0 or resnet_fraud_risk > 50.0) and not ai_res["is_ai_generated"]:
        max_prob = max(neural_fraud_risk, resnet_fraud_risk)
        reasons.append({
            "type": "AI_GENERATION",
            "severity": "HIGH",
            "title": "Neural Manipulation Alert",
            "explanation": f"Neural classifier detected a {max_prob:.1f}% probability of text alteration or inpainting."
        })

    fraud_percentage = round(max(0.0, min(99.0, raw_fraud_pct)), 1)

    if not reasons:
        verdict = "GENUINE"
        badge_color = "emerald"
        status_label = "Verified Genuine Document"
        summary = "No fraud indicators found. The document has matching line items, authentic paper grain, and is not a duplicate."
    elif fraud_percentage >= 70.0:
        verdict = "HIGH_FRAUD_RISK"
        badge_color = "red"
        status_label = "Fake / Highly Fraudulent"
        summary = f"Flagged with {fraud_percentage}% fraud probability. Found {len(reasons)} critical reason(s) to report this document as fraudulent."
    else:
        verdict = "SUSPICIOUS"
        badge_color = "amber"
        status_label = "Suspicious Document"
        summary = f"Flagged with {fraud_percentage}% fraud probability. Moderate irregularities detected; manual accounting review recommended."

    # Record scan
    heuristic_engine.record_scan(
        filename=filename,
        inv_num=parsed.get("invoice_number") or "N/A",
        sha256=audit_res["sha256"],
        phash=audit_res["phash"],
        total=parsed.get("total") or 0.0,
        score=100.0 - fraud_percentage
    )

    return {
        "filename": filename,
        "fraud_percentage": fraud_percentage,
        "verdict": verdict,
        "status_label": status_label,
        "badge_color": badge_color,
        "summary": summary,
        "reasons": reasons,
        "parsed_fields": parsed,
        "byte_offset_diagnostics": diagnostics.get("byte_offset_diagnostics", []),
        "whitelist_status": diagnostics.get("whitelist_status"),
        "image_original": mat_to_base64(img_bgr),
        "image_highlighted": mat_to_base64(gradcam_overlay)
    }

@app.get("/", response_class=HTMLResponse)
def serve_dashboard():
    with open(os.path.join(BASE_DIR, "src", "backend", "dashboard.html"), "r", encoding="utf-8") as f:
        return f.read()

if __name__ == "__main__":
    import uvicorn
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host=host, port=port)
