import os
from typing import Dict, Any, Optional
from PIL import Image
from PIL.ExifTags import TAGS

from src.forensics.jumbf_byte_parser import JUMBFByteParser, ParsedOffsetsResult, DiagnosticItem
from src.forensics.approved_exif_registry import ApprovedExifRegistry, WhitelistedProfile

class MetadataGuard:
    """
    Granular Metadata Guard combining binary byte offset inspection and approved EXIF registry.
    Detects full and partial AI provenance markers (e.g., truncated JUMBF blocks),
    validates legitimate scanner/camera signatures, and exposes spoofed metadata.
    """
    def __init__(self, parser: Optional[JUMBFByteParser] = None, registry: Optional[ApprovedExifRegistry] = None):
        self.parser = parser or JUMBFByteParser()
        self.registry = registry or ApprovedExifRegistry()

    def evaluate(self, image_path: str) -> Dict[str, Any]:
        """
        Evaluates document image for C2PA/EXIF integrity, byte offsets, and whitelist provenance.
        """
        offsets: ParsedOffsetsResult = self.parser.parse(image_path)

        # Fallback EXIF extraction via PIL if binary IFD parsing was partial
        if not offsets.exif and os.path.exists(image_path):
            self._supplement_exif_with_pil(image_path, offsets)

        exif_status: Optional[WhitelistedProfile] = self.registry.evaluate_exif(offsets.exif)

        # Cross-check anti-spoofing: Whitelist claimed + AI tag or Truncation detected
        if exif_status and not exif_status.is_spoofed_attempt:
            if offsets.has_ai_tags or offsets.truncated:
                offsets.details.append(DiagnosticItem(
                    offset=0,
                    offset_hex="0x00000000",
                    length=0,
                    tag_type="SPOOFED_SCANNER_METADATA",
                    context=f"Document claims legitimate hardware capture ('{exif_status.make} {exif_status.model}') but contains AI provenance tags or truncated JUMBF blocks (Anti-forensic Spoofing Detected)",
                    severity="CRITICAL"
                ))

        if exif_status and exif_status.is_spoofed_attempt:
            offsets.details.append(DiagnosticItem(
                offset=0,
                offset_hex="0x00000000",
                length=0,
                tag_type="SPOOFED_SCANNER_METADATA",
                context=exif_status.spoof_reason or "Typo-squatted / spoofed hardware EXIF tag detected",
                severity="CRITICAL"
            ))

        risk_score = self._score(offsets, exif_status)
        is_ai = offsets.has_ai_tags or (exif_status and exif_status.is_spoofed_attempt)

        return {
            "is_ai": is_ai,
            "is_tampered_truncated": offsets.truncated,
            "byte_offset_diagnostics": [item.to_dict() for item in offsets.details],
            "whitelist_status": exif_status.to_dict() if exif_status else None,
            "risk_score": risk_score,
            "file_format": offsets.format,
            "file_size": offsets.file_size,
            "markers_or_chunks": offsets.markers_or_chunks,
            "jumbf_boxes": offsets.jumbf_boxes
        }

    def _score(self, offsets: ParsedOffsetsResult, exif_status: Optional[WhitelistedProfile]) -> float:
        """
        Calculates calibrated metadata risk score:
        +40 if AI tag found
        +30 if truncated JUMBF found
        -20 if verified whitelist
        +30 to +40 if spoofed scanner metadata
        """
        score = 0.0

        if offsets.has_ai_tags:
            score += 40.0

        if offsets.truncated:
            score += 30.0

        if exif_status:
            if exif_status.is_spoofed_attempt:
                score += 40.0
            elif offsets.has_ai_tags or offsets.truncated:
                score += 30.0  # Claimed whitelist but actually AI/tampered
            else:
                score -= 20.0  # Genuine approved device/software

        return max(0.0, min(100.0, score))

    def _supplement_exif_with_pil(self, image_path: str, offsets: ParsedOffsetsResult):
        """
        Supplements EXIF dictionary using PIL for standard image formats.
        """
        try:
            with Image.open(image_path) as img:
                exif = img.getexif()
                if exif:
                    for tag_id, value in exif.items():
                        tag_name = TAGS.get(tag_id, str(tag_id))
                        if tag_name in ["Make", "Model", "Software"]:
                            offsets.exif[tag_name] = str(value)
                # Check info for software
                if "Software" not in offsets.exif and img.info:
                    if "Software" in img.info:
                        offsets.exif["Software"] = str(img.info["Software"])
        except Exception:
            pass
