import re
from dataclasses import dataclass, asdict
from typing import Dict, Tuple, Optional, Any, List

def levenshtein_distance(s1: str, s2: str) -> int:
    """
    Computes the Levenshtein edit distance between two strings.
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]

@dataclass
class WhitelistedProfile:
    vendor: str
    category: str  # "Scanner", "Mobile Camera", "Enterprise Document System", "Camera"
    make: str
    model: str
    software: str
    confidence: float = 1.0
    is_spoofed_attempt: bool = False
    spoof_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

# Master registry of approved (Make, Model, Software) hardware & document engines
APPROVED: Dict[Tuple[str, str, str], Dict[str, str]] = {
    # Canon Physical Scanners
    ("Canon", "CanoScan LiDE 300", "Canon IJ Scan Utility"): {"vendor": "Canon", "category": "Scanner"},
    ("Canon", "CanoScan LiDE 400", "Canon IJ Scan Utility"): {"vendor": "Canon", "category": "Scanner"},
    ("Canon", "CanoScan LiDE 220", "Canon IJ Scan Utility"): {"vendor": "Canon", "category": "Scanner"},
    ("Canon", "CanoScan 9000F", "Canon MP Navigator EX"): {"vendor": "Canon", "category": "Scanner"},
    ("Canon", "imageFORMULA DR-C225", "CaptureOnTouch"): {"vendor": "Canon", "category": "Scanner"},
    ("Canon", "imageFORMULA P-215II", "CaptureOnTouch"): {"vendor": "Canon", "category": "Scanner"},
    ("Canon", "Canon MF Scan", "Canon MF Scan Utility"): {"vendor": "Canon", "category": "Scanner"},

    # Epson Physical Scanners
    ("Epson", "Perfection V39", "Epson Scan"): {"vendor": "Epson", "category": "Scanner"},
    ("Epson", "Perfection V39 II", "Epson Scan 2"): {"vendor": "Epson", "category": "Scanner"},
    ("Epson", "Perfection V600 Photo", "Epson Scan"): {"vendor": "Epson", "category": "Scanner"},
    ("Epson", "WorkForce DS-530", "Document Capture Pro"): {"vendor": "Epson", "category": "Scanner"},
    ("Epson", "WorkForce ES-400", "Epson Scan 2"): {"vendor": "Epson", "category": "Scanner"},
    ("Epson", "EPSON Perfection", "Epson Scan"): {"vendor": "Epson", "category": "Scanner"},

    # HP Physical Scanners & MFPs
    ("HP", "ScanJet Pro 2500 f1", "HP Scan"): {"vendor": "HP", "category": "Scanner"},
    ("HP", "ScanJet Pro 3000 s4", "HP Scan"): {"vendor": "HP", "category": "Scanner"},
    ("HP", "ScanJet Enterprise Flow", "HP EveryPage"): {"vendor": "HP", "category": "Scanner"},
    ("HP", "HP LaserJet MFP", "HP Smart"): {"vendor": "HP", "category": "Scanner"},
    ("HP", "HP OfficeJet Pro", "HP Smart"): {"vendor": "HP", "category": "Scanner"},

    # Fujitsu / PFU / Ricoh Document Scanners
    ("Fujitsu", "ScanSnap iX1500", "ScanSnap Home"): {"vendor": "Fujitsu", "category": "Scanner"},
    ("Fujitsu", "ScanSnap iX1600", "ScanSnap Home"): {"vendor": "Fujitsu", "category": "Scanner"},
    ("Fujitsu", "ScanSnap iX500", "ScanSnap Manager"): {"vendor": "Fujitsu", "category": "Scanner"},
    ("PFU Limited", "fi-7160", "PaperStream IP"): {"vendor": "Fujitsu", "category": "Scanner"},
    ("Ricoh", "ScanSnap iX1600", "ScanSnap Home"): {"vendor": "Ricoh", "category": "Scanner"},

    # Brother Scanners & MFPs
    ("Brother", "ADS-1700W", "Brother iPrint&Scan"): {"vendor": "Brother", "category": "Scanner"},
    ("Brother", "ADS-2700W", "Brother ControlCenter4"): {"vendor": "Brother", "category": "Scanner"},
    ("Brother", "MFC-L2750DW", "Brother iPrint&Scan"): {"vendor": "Brother", "category": "Scanner"},

    # Xerox Scanners
    ("Xerox", "DocuMate 3125", "Visioneer OneTouch"): {"vendor": "Xerox", "category": "Scanner"},
    ("Xerox", "DocuMate 6440", "Visioneer Acuity"): {"vendor": "Xerox", "category": "Scanner"},

    # Mobile Document Capture (Smartphones)
    ("Apple", "iPhone 11", "Apple Camera"): {"vendor": "Apple", "category": "Mobile Camera"},
    ("Apple", "iPhone 12", "Apple Camera"): {"vendor": "Apple", "category": "Mobile Camera"},
    ("Apple", "iPhone 13", "Apple Camera"): {"vendor": "Apple", "category": "Mobile Camera"},
    ("Apple", "iPhone 14 Pro", "Apple Camera"): {"vendor": "Apple", "category": "Mobile Camera"},
    ("Apple", "iPhone 14", "Apple Camera"): {"vendor": "Apple", "category": "Mobile Camera"},
    ("Apple", "iPhone 15 Pro", "Apple Camera"): {"vendor": "Apple", "category": "Mobile Camera"},
    ("Apple", "iPhone 15", "Apple Camera"): {"vendor": "Apple", "category": "Mobile Camera"},
    ("Apple", "iPhone 16 Pro", "Apple Camera"): {"vendor": "Apple", "category": "Mobile Camera"},
    ("Apple", "iPad", "Apple Camera"): {"vendor": "Apple", "category": "Mobile Camera"},

    ("Samsung", "Galaxy S21", "Samsung Camera"): {"vendor": "Samsung", "category": "Mobile Camera"},
    ("Samsung", "Galaxy S22", "Samsung Camera"): {"vendor": "Samsung", "category": "Mobile Camera"},
    ("Samsung", "Galaxy S23", "Samsung Camera"): {"vendor": "Samsung", "category": "Mobile Camera"},
    ("Samsung", "Galaxy S24", "Samsung Camera"): {"vendor": "Samsung", "category": "Mobile Camera"},

    ("Google", "Pixel 6", "Google Camera"): {"vendor": "Google", "category": "Mobile Camera"},
    ("Google", "Pixel 7", "Google Camera"): {"vendor": "Google", "category": "Mobile Camera"},
    ("Google", "Pixel 8", "Google Camera"): {"vendor": "Google", "category": "Mobile Camera"},

    # Legitimate Mobile Document Scanner Apps
    ("Adobe", "Adobe Scan Mobile", "Adobe Scan"): {"vendor": "Adobe", "category": "Mobile Scanner App"},
    ("INTSIG", "CamScanner", "CamScanner App"): {"vendor": "CamScanner", "category": "Mobile Scanner App"},
    ("Microsoft", "Microsoft Lens", "Office Lens iOS"): {"vendor": "Microsoft", "category": "Mobile Scanner App"},
    ("Microsoft", "Microsoft Lens", "Office Lens Android"): {"vendor": "Microsoft", "category": "Mobile Scanner App"},
    ("The Grizzly Labs", "Genius Scan", "Genius Scan App"): {"vendor": "Genius Scan", "category": "Mobile Scanner App"},

    # Accounting & Enterprise Document Engines
    ("SAP", "Crystal Reports", "SAP ERP Document Exporter"): {"vendor": "SAP", "category": "Enterprise Document System"},
    ("Intuit", "QuickBooks Invoice Engine", "QuickBooks Desktop"): {"vendor": "Intuit", "category": "Enterprise Document System"},
    ("Oracle", "BI Publisher", "Oracle Reports"): {"vendor": "Oracle", "category": "Enterprise Document System"}
}

KNOWN_VENDORS = [
    "canon", "epson", "hp", "hewlett-packard", "fujitsu", "pfu limited",
    "brother", "xerox", "apple", "samsung", "google", "adobe", "intsig",
    "microsoft", "the grizzly labs", "sap", "intuit", "oracle", "ricoh"
]

class ApprovedExifRegistry:
    """
    Registry of approved camera, scanner, and document software EXIF signatures.
    Eliminates false positives on genuine receipts and catches spoofed tags using Levenshtein distance.
    """
    def __init__(self, custom_approved: Optional[Dict[Tuple[str, str, str], Dict[str, str]]] = None):
        self.registry = dict(APPROVED)
        if custom_approved:
            self.registry.update(custom_approved)

        # Build normalized lookup table: (make_lower, model_lower, software_lower) -> profile
        self.normalized_lookup = {}
        for (make, model, software), meta in self.registry.items():
            k = (make.strip().lower(), model.strip().lower(), software.strip().lower())
            self.normalized_lookup[k] = {
                "vendor": meta["vendor"],
                "category": meta["category"],
                "make": make,
                "model": model,
                "software": software
            }

    def evaluate_exif(self, exif_dict: Optional[Dict[str, Any]]) -> Optional[WhitelistedProfile]:
        """
        Evaluates EXIF dictionary:
        1. Normalizes strings.
        2. Exact lookup against registry.
        3. Prefix / flexible software version matching.
        4. Fuzzy matching (Levenshtein <= 2) to catch spoofed tags like 'Can0n' or 'App1e'.
        """
        if not exif_dict:
            return None

        make_raw = str(exif_dict.get("Make", "") or "").strip()
        model_raw = str(exif_dict.get("Model", "") or "").strip()
        software_raw = str(exif_dict.get("Software", "") or "").strip()

        if not make_raw and not model_raw and not software_raw:
            return None

        make_norm = make_raw.lower()
        model_norm = model_raw.lower()
        software_norm = software_raw.lower()

        # 1. Exact Match
        exact_key = (make_norm, model_norm, software_norm)
        if exact_key in self.normalized_lookup:
            meta = self.normalized_lookup[exact_key]
            return WhitelistedProfile(
                vendor=meta["vendor"],
                category=meta["category"],
                make=make_raw,
                model=model_raw,
                software=software_raw,
                confidence=1.0,
                is_spoofed_attempt=False
            )

        # 2. Fuzzy / Spoof Check (Levenshtein <= 2 on Make)
        # Catch spoofed tags such as 'Can0n', 'App1e', 'Eps0n'
        for vendor in KNOWN_VENDORS:
            if make_norm and make_norm != vendor:
                dist = levenshtein_distance(make_norm, vendor)
                # If distance is 1 or 2, flag as spoofed attempt
                if 1 <= dist <= 2 and len(make_norm) >= 3:
                    return WhitelistedProfile(
                        vendor=vendor.title(),
                        category="Spoofed Tag",
                        make=make_raw,
                        model=model_raw,
                        software=software_raw,
                        confidence=0.0,
                        is_spoofed_attempt=True,
                        spoof_reason=f"Make '{make_raw}' has Levenshtein distance {dist} from approved vendor '{vendor.title()}' (Spoofing Detected)"
                    )

        # 3. Flexible Match (e.g., matching known Make + Model with varying Software version)
        for (reg_make, reg_model, reg_soft), meta in self.normalized_lookup.items():
            make_matches = (reg_make in make_norm) or (make_norm in reg_make)
            model_matches = (reg_model in model_norm) or (model_norm in reg_model)
            soft_matches = (reg_soft in software_norm) or (software_norm in reg_soft) or (not software_norm and not reg_soft)

            if make_matches and model_matches and (soft_matches or not software_norm):
                return WhitelistedProfile(
                    vendor=meta["vendor"],
                    category=meta["category"],
                    make=make_raw,
                    model=model_raw,
                    software=software_raw or meta["software"],
                    confidence=0.90,
                    is_spoofed_attempt=False
                )

        # 4. Standalone Vendor Scanner / Camera Match (e.g. Make: "Canon", Software: "Canon IJ Scan Utility")
        for (reg_make, reg_model, reg_soft), meta in self.normalized_lookup.items():
            if make_norm == reg_make and (reg_soft and reg_soft in software_norm):
                return WhitelistedProfile(
                    vendor=meta["vendor"],
                    category=meta["category"],
                    make=make_raw,
                    model=model_raw or meta["model"],
                    software=software_raw,
                    confidence=0.85,
                    is_spoofed_attempt=False
                )

        # 5. Mobile Scan apps (Adobe Scan, CamScanner) where Make might be absent or device-specific
        if "adobe scan" in software_norm:
            return WhitelistedProfile(
                vendor="Adobe",
                category="Mobile Scanner App",
                make=make_raw or "Mobile Device",
                model=model_raw or "Scanner App",
                software=software_raw,
                confidence=0.95,
                is_spoofed_attempt=False
            )
        if "camscanner" in software_norm or "intsig" in software_norm:
            return WhitelistedProfile(
                vendor="CamScanner",
                category="Mobile Scanner App",
                make=make_raw or "Mobile Device",
                model=model_raw or "Scanner App",
                software=software_raw,
                confidence=0.95,
                is_spoofed_attempt=False
            )

        return None
