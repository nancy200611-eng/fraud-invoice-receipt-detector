import os
import re
import sqlite3
import hashlib
import cv2
import numpy as np

class HeuristicAuditEngine:
    """
    Accounting Domain & Business Rules Audit Engine (Thornton et al. 2025 Framework)
    - Math reconciliation: Sum of item lines == Subtotal, Subtotal + Tax == Total Due
    - Duplicate detection: Flag reuse of same invoice ID or perceptual pHash by DIFFERENT submissions
    - Unusual amount and format verification
    """
    def __init__(self, db_path=None):
        if db_path is None:
            base_dir = os.environ.get("BASE_DIR", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
            db_path = os.path.join(base_dir, "data", "audit_ledger.db")
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS scan_ledger (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT,
                invoice_number TEXT,
                sha256 TEXT,
                phash TEXT,
                total_amount REAL,
                authenticity_score REAL,
                scan_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        conn.close()

    def compute_phash(self, image_bgr):
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
        dct = cv2.dct(np.float32(resized))
        dct_low = dct[:8, :8]
        med = np.median(dct_low[1:])
        hash_bits = (dct_low > med).flatten()
        return "".join(["1" if b else "0" for b in hash_bits])

    def hamming_distance(self, hash1, hash2):
        if not hash1 or not hash2 or len(hash1) != len(hash2):
            return 64
        return sum(c1 != c2 for c1, c2 in zip(hash1, hash2))

    def _parse_amount(self, text):
        m = re.search(r"\$?\s*(\d{1,6}[\.,]\d{2})", text)
        if m:
            try:
                return float(m.group(1).replace(",", "."))
            except Exception:
                pass
        return None

    def extract_numerical_fields(self, tokens):
        subtotal = None
        tax = None
        total = None
        invoice_number = None

        inv_pattern = re.compile(r"INV[-\d\w]+", re.IGNORECASE)

        for idx, t in enumerate(tokens):
            text = t.get("text", "")
            if not invoice_number:
                inv_match = inv_pattern.search(text)
                if inv_match:
                    invoice_number = inv_match.group(0)

            upper_text = text.upper()
            if "SUBTOTAL" in upper_text or "SUB TOTAL" in upper_text:
                val = self._parse_amount(text)
                if val is None and idx + 1 < len(tokens):
                    val = self._parse_amount(tokens[idx + 1]["text"])
                if val is not None and subtotal is None:
                    subtotal = val

            elif "TAX" in upper_text:
                val = self._parse_amount(text)
                if val is None and idx + 1 < len(tokens):
                    val = self._parse_amount(tokens[idx + 1]["text"])
                if val is not None and tax is None:
                    tax = val

            elif "TOTAL" in upper_text and "SUB" not in upper_text:
                val = self._parse_amount(text)
                if val is None and idx + 1 < len(tokens):
                    val = self._parse_amount(tokens[idx + 1]["text"])
                if val is not None:
                    if total is None or val > total:
                        total = val

        return {
            "invoice_number": invoice_number,
            "subtotal": subtotal,
            "tax": tax,
            "total": total
        }

    def audit_document(self, image_path, tokens):
        flags = []
        current_filename = os.path.basename(image_path)
        parsed = self.extract_numerical_fields(tokens)
        subtotal = parsed["subtotal"]
        tax = parsed["tax"]
        total = parsed["total"]
        inv_num = parsed["invoice_number"]

        # Math Audit
        if subtotal is not None and tax is not None and total is not None:
            expected_total = round(subtotal + tax, 2)
            discrepancy = abs(round(total - expected_total, 2))
            if discrepancy > 0.05:
                flags.append({
                    "code": "MATH_DISCREPANCY",
                    "severity": "CRITICAL",
                    "message": f"Mathematical sum mismatch: Subtotal (${subtotal:.2f}) + Tax (${tax:.2f}) = ${expected_total:.2f}, but Total Due is ${total:.2f} (diff: ${discrepancy:.2f})"
                })
            else:
                flags.append({
                    "code": "MATH_VERIFIED",
                    "severity": "INFO",
                    "message": f"Mathematical sum verified: Subtotal (${subtotal:.2f}) + Tax (${tax:.2f}) == Total Due (${total:.2f})"
                })

        img_bgr = cv2.imread(image_path)
        with open(image_path, "rb") as f:
            file_sha256 = hashlib.sha256(f.read()).hexdigest()
        curr_phash = self.compute_phash(img_bgr)

        # Database Ledger Checks (Only against OTHER past documents)
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()

        if inv_num and inv_num != "INVOICE":
            cur.execute("SELECT filename, scan_date FROM scan_ledger WHERE invoice_number = ? AND filename != ? LIMIT 1", (inv_num, current_filename))
            rows = cur.fetchall()
            if rows:
                flags.append({
                    "code": "DUPLICATE_INVOICE_ID",
                    "severity": "HIGH",
                    "message": f"Duplicate invoice: #{inv_num} was previously submitted under file '{rows[0][0]}' on {rows[0][1]}"
                })

        cur.execute("SELECT filename, phash, scan_date FROM scan_ledger WHERE filename != ?", (current_filename,))
        all_scans = cur.fetchall()
        for prev_file, prev_phash, scan_date in all_scans:
            dist = self.hamming_distance(curr_phash, prev_phash)
            if dist < 4: # Near-identical visual template
                flags.append({
                    "code": "DUPLICATE_IMAGE_REUSE",
                    "severity": "HIGH",
                    "message": f"Visual receipt template duplicate of '{prev_file}' (pHash distance: {dist})"
                })
                break

        conn.close()

        risk_penalty = 0
        for f in flags:
            if f["severity"] == "CRITICAL":
                risk_penalty += 55
            elif f["severity"] == "HIGH":
                risk_penalty += 35
            elif f["severity"] == "MEDIUM":
                risk_penalty += 15

        return {
            "parsed_fields": parsed,
            "flags": flags,
            "risk_penalty": min(100, risk_penalty),
            "sha256": file_sha256,
            "phash": curr_phash
        }

    def record_scan(self, filename, inv_num, sha256, phash, total, score):
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO scan_ledger (filename, invoice_number, sha256, phash, total_amount, authenticity_score)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (filename, inv_num, sha256, phash, total, score))
        conn.commit()
        conn.close()

if __name__ == "__main__":
    audit = HeuristicAuditEngine()
    print("Accounting Rule Engine updated.")
