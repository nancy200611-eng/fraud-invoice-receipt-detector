import os
import io
import sys
import base64
import cv2
import numpy as np
import gradio as gr
from PIL import Image

# Ensure project root is in python path
BASE_DIR = os.environ.get(
    "BASE_DIR",
    os.path.dirname(os.path.abspath(__file__))
)
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.backend.main import process_document

def b64_to_rgb(b64_str):
    if not b64_str:
        return None
    if "," in b64_str:
        b64_str = b64_str.split(",")[1]
    decoded = base64.b64decode(b64_str)
    nparr = np.frombuffer(decoded, np.uint8)
    bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

def analyze_receipt(image_path):
    if not image_path or not os.path.exists(image_path):
        return None, "<div style='color:red;'>?? Please upload an invoice or receipt image.</div>", ""

    filename = os.path.basename(image_path)
    res = process_document(image_path, filename)

    # Process heatmap overlay image
    overlay_img = b64_to_rgb(res.get("image_highlighted"))

    # Determine badge styling
    verdict = res.get("verdict", "UNKNOWN")
    score = res.get("fraud_percentage", 0.0)
    
    if verdict == "GENUINE":
        color = "#10b981"
        status_icon = "?"
    elif verdict == "HIGH_FRAUD_RISK":
        color = "#ef4444"
        status_icon = "??"
    else:
        color = "#f59e0b"
        status_icon = "??"

    summary_html = f"""
    <div style="background-color: {color}15; border-left: 6px solid {color}; padding: 16px; border-radius: 8px; margin-bottom: 16px;">
        <h2 style="color: {color}; margin: 0 0 8px 0;">{status_icon} {res.get('status_label', verdict)}</h2>
        <p style="font-size: 1.1em; font-weight: 600; margin: 0 0 6px 0;">Estimated Fraud Risk: <span style="color: {color}; font-size: 1.3em;">{score:.1f}%</span></p>
        <p style="color: #4b5563; margin: 0;">{res.get('summary', '')}</p>
    </div>
    """

    # Reasons list markdown
    reasons = res.get("reasons", [])
    reasons_md = "### ?? Forensic Findings & Reasons\n"
    if not reasons:
        reasons_md += "? No suspicious tampering or anomalies detected. The document passed all verification checks.\n"
    else:
        for r in reasons:
            sev = r.get("severity", "INFO")
            title = r.get("title", "")
            exp = r.get("explanation", "")
            reasons_md += f"- **[{sev}] {title}**: {exp}\n"

    # Parsed fields
    parsed = res.get("parsed_fields", {})
    reasons_md += "\n### ?? Extracted Document Details\n"
    reasons_md += f"- **Invoice Number**: {parsed.get('invoice_number') or 'N/A'}\n"
    reasons_md += f"- **Subtotal**: {parsed.get('subtotal') or 'N/A'}\n"
    reasons_md += f"- **Tax**: {parsed.get('tax') or 'N/A'}\n"
    reasons_md += f"- **Total**: {parsed.get('total') or 'N/A'}\n"

    return overlay_img, summary_html, reasons_md

# Example images
examples = []
sample_raw = os.path.join(BASE_DIR, "data", "raw", "invoice_0001.jpg")
sample_fake = os.path.join(BASE_DIR, "data", "manipulated", "images", "fake_invoice_0001.jpg")
if os.path.exists(sample_raw):
    examples.append([sample_raw])
if os.path.exists(sample_fake):
    examples.append([sample_fake])

with gr.Blocks(title="Receipt & Invoice Fraud Detector") as demo:
    gr.Markdown("""
    # ??? Receipt & Invoice Fraud Detection Platform
    Upload any receipt or invoice to run **Deep Learning (ResNet-18 + Grad-CAM)**, **Digital Forensics (ELA & SRM)**, **EasyOCR extraction**, and **Business Math Ledger Auditing**.
    """)

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="filepath", label="Upload Invoice / Receipt Image")
            submit_btn = gr.Button("?? Analyze Document", variant="primary", size="lg")
            
            if examples:
                gr.Examples(
                    examples=examples,
                    inputs=input_image,
                    label="Test Examples (Click to test)"
                )

        with gr.Column():
            verdict_output = gr.HTML(label="Fraud Verdict")
            overlay_output = gr.Image(label="Forensic Grad-CAM Tampering Heatmap")
            findings_output = gr.Markdown(label="Audit Report")

    submit_btn.click(
        fn=analyze_receipt,
        inputs=[input_image],
        outputs=[overlay_output, verdict_output, findings_output]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
