import os
import glob
import time
import numpy as np
from src.backend.main import process_document

def run_performance_benchmark():
    real_receipts = glob.glob(r"D:\SOIRE\ICDAR-2019-SROIE-master\data\img\*.jpg")[:10]
    real_invoices = glob.glob(r"D:\Receipt-Invoice-fraud-detection\data\real\invoice\*.jpg")[:10]
    fake_invoices = glob.glob(r"D:\Receipt-Invoice-fraud-detection\data\fake\invoice\*.jpg")[:10]
    fake_receipts = glob.glob(r"D:\Receipt-Invoice-fraud-detection\data\fake\receipt\*.jpg")[:10]
    chatgpt_sample = r"E:\fraud Invoice&Receipt Detector\data\uploads\ChatGPT Image Sep 13, 2026, 06_00_24 PM.png"

    all_real = [(p, "GENUINE") for p in real_receipts + real_invoices]
    all_fake = [(p, "FAKE") for p in fake_invoices + fake_receipts]
    if os.path.exists(chatgpt_sample):
        all_fake.append((chatgpt_sample, "FAKE"))

    print(f"Running Performance Benchmark across {len(all_real)} Genuine and {len(all_fake)} Fraudulent documents...")

    tp = 0 # True Positive: Fake correctly flagged as Fake
    fp = 0 # False Positive: Real incorrectly flagged as Fake
    tn = 0 # True Negative: Real correctly identified as Genuine/Auditable
    fn = 0 # False Negative: Fake missed as Genuine

    latencies_full = []
    latencies_fast = []

    # Evaluate Real
    for path, expected in all_real:
        t0 = time.time()
        res = process_document(path, os.path.basename(path))
        dt = (time.time() - t0) * 1000
        latencies_full.append(dt)

        if res["verdict"] == "HIGH_FRAUD_RISK":
            fp += 1
        else:
            tn += 1

    # Evaluate Fake
    for path, expected in all_fake:
        t0 = time.time()
        res = process_document(path, os.path.basename(path))
        dt = (time.time() - t0) * 1000

        if "C2PA" in res["status_label"] or "Fast-Path" in res["status_label"]:
            latencies_fast.append(dt)
        else:
            latencies_full.append(dt)

        if res["verdict"] in ["HIGH_FRAUD_RISK", "CRITICAL_FRAUD"]:
            tp += 1
        elif res["fraud_percentage"] >= 50.0:
            tp += 1
        else:
            fn += 1

    total = len(all_real) + len(all_fake)
    accuracy = (tp + tn) / total * 100.0
    precision = tp / (tp + fp or 1) * 100.0
    recall = tp / (tp + fn or 1) * 100.0
    f1 = 2 * (precision * recall) / (precision + recall or 1)
    fpr = fp / (fp + tn or 1) * 100.0

    print("\n================ SYSTEM PERFORMANCE REPORT ================")
    print(f"Total Evaluated: {total} documents ({len(all_real)} Genuine, {len(all_fake)} Fraudulent)")
    print(f"True Positives (Fraud Caught):    {tp}")
    print(f"True Negatives (Genuine Passed):  {tn}")
    print(f"False Positives (Genuine Flagged):{fp}")
    print(f"False Negatives (Fraud Missed):   {fn}")
    print("-----------------------------------------------------------")
    print(f"Classification Accuracy:          {accuracy:.1f}%")
    print(f"Fraud Detection Recall:           {recall:.1f}%")
    print(f"Precision:                        {precision:.1f}%")
    print(f"F1-Score:                         {f1:.1f}%")
    print(f"False Positive Rate (FPR):        {fpr:.1f}%")
    print("-----------------------------------------------------------")
    print(f"Fast-Path Latency (C2PA/EXIF):     {np.mean(latencies_fast):.2f} ms" if latencies_fast else "Fast-Path: N/A")
    print(f"Average Full Pipeline Latency:    {np.mean(latencies_full)/1000:.2f} seconds (CPU)")
    print("===========================================================")

if __name__ == "__main__":
    run_performance_benchmark()
