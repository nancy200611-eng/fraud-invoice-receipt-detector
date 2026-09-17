import os
import requests
import pytest

API_BASE = "http://127.0.0.1:8000"

def test_api_health():
    resp = requests.get(f"{API_BASE}/api/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ready"

def test_genuine_sample_scan():
    resp = requests.get(f"{API_BASE}/api/scan-sample/genuine")
    assert resp.status_code == 200
    data = resp.json()
    assert "fraud_percentage" in data
    assert "reasons" in data
    assert "verdict" in data
    assert data["verdict"] in ["GENUINE", "SUSPICIOUS"]
    print(f"\n[Genuine Sample] Fraud: {data['fraud_percentage']}% | Verdict: {data['verdict']}")

def test_fake_sample_scan():
    resp = requests.get(f"{API_BASE}/api/scan-sample/fake")
    assert resp.status_code == 200
    data = resp.json()
    assert "fraud_percentage" in data
    assert "reasons" in data
    assert len(data["reasons"]) >= 1
    assert data["fraud_percentage"] >= 70.0
    print(f"\n[Fake Sample] Fraud: {data['fraud_percentage']}% | Verdict: {data['verdict']}")
    for r in data["reasons"]:
        print(f" - [{r['title']}]: {r['explanation']}")

def test_universal_ai_detector():
    from src.forensics.ai_generator_detector import AIGeneratorDetector
    detector = AIGeneratorDetector()
    assert "Midjourney" in detector.ai_engine_signatures
    assert "Stable Diffusion / SDXL" in detector.ai_engine_signatures
    assert "Flux" in detector.ai_engine_signatures
    assert "Online Receipt Builder" in detector.ai_engine_signatures

def test_provenance_seal():
    from src.forensics.robustness import ShieldRobustnessEngine
    shield = ShieldRobustnessEngine()
    data = b"Invoice #1234 Total: $500.00"
    seal = shield.sign_document(data)
    assert shield.verify_provenance(data, seal) is True
