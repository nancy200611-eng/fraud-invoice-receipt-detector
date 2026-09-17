import os
import hmac
import hashlib
import cv2
import numpy as np
from sklearn.ensemble import IsolationForest

class ShieldRobustnessEngine:
    """
    Phase 5: Anti-Gravity Shield & Resilience Engine
    - Adversarial / Perturbation Robustness testing
    - Cryptographic provenance and tamper-evident signatures
    - Unsupervised Anomaly Detection on feature vectors
    """
    def __init__(self):
        self.anomaly_detector = IsolationForest(contamination=0.15, random_state=42)
        self.fitted = False

    def simulate_perturbations(self, image_bgr):
        """
        Generates realistic post-processing perturbations:
        1. Heavy JPEG re-compression (Q=40)
        2. Scanning/printer blur (Gaussian kernel)
        3. Minor rotation (-2 degrees)
        """
        # 1. JPEG compression
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 40]
        _, enc = cv2.imencode('.jpg', image_bgr, encode_param)
        jpeg_perturbed = cv2.imdecode(enc, 1)

        # 2. Scanner blur
        blurred = cv2.GaussianBlur(image_bgr, (5, 5), 1.2)

        # 3. Skew / Rotation
        h, w = image_bgr.shape[:2]
        center = (w // 2, h // 2)
        rot_matrix = cv2.getRotationMatrix2D(center, -2.0, 1.0)
        rotated = cv2.warpAffine(image_bgr, rot_matrix, (w, h), borderValue=(255, 255, 255))

        return {
            "jpeg_compressed": jpeg_perturbed,
            "scanner_blurred": blurred,
            "skew_rotated": rotated
        }

    def sign_document(self, image_bytes, secret_key=b"antigravity-secret-key-2026"):
        """
        Computes HMAC-SHA256 digital provenance seal.
        """
        return hmac.new(secret_key, image_bytes, hashlib.sha256).hexdigest()

    def verify_provenance(self, image_bytes, signature, secret_key=b"antigravity-secret-key-2026"):
        expected = self.sign_document(image_bytes, secret_key)
        return hmac.compare_digest(expected, signature)

    def fit_anomaly_detector(self, feature_matrix):
        """
        Fits IsolationForest on genuine document feature embeddings.
        """
        self.anomaly_detector.fit(feature_matrix)
        self.fitted = True

    def detect_anomaly(self, feature_vector):
        """
        Returns anomaly score [-1: anomaly, 1: normal]
        """
        if not self.fitted:
            return {"is_anomaly": False, "score": 0.5}
        pred = self.anomaly_detector.predict([feature_vector])[0]
        decision = self.anomaly_detector.decision_function([feature_vector])[0]
        return {
            "is_anomaly": bool(pred == -1),
            "anomaly_score": round(float(decision), 4)
        }

if __name__ == "__main__":
    shield = ShieldRobustnessEngine()
    sample = r"E:\fraud Invoice&Receipt Detector\data\raw\invoice_0001.jpg"
    img = cv2.imread(sample)
    perts = shield.simulate_perturbations(img)
    print("Perturbations simulated successfully:", list(perts.keys()))
    with open(sample, "rb") as f:
        data = f.read()
    sig = shield.sign_document(data)
    is_valid = shield.verify_provenance(data, sig)
    print(f"Digital Provenance Seal: {sig[:16]}... Valid: {is_valid}")
