import os
import re
from typing import Tuple, Optional, Dict, Any
from PIL import Image
from PIL.ExifTags import TAGS

from src.forensics.jumbf_byte_parser import JUMBFByteParser
from src.forensics.approved_exif_registry import ApprovedExifRegistry
from src.forensics.metadata_guard import MetadataGuard

class FastPathResult(tuple):
    """
    Backward-compatible tuple (is_ai, reason_title, reason_explanation)
    with extended diagnostics accessible via .diagnostics attribute or dict lookup.
    """
    def __new__(cls, is_ai: bool, reason_title: Optional[str], reason_explanation: Optional[str], diagnostics: Optional[Dict[str, Any]] = None):
        obj = super().__new__(cls, (is_ai, reason_title, reason_explanation))
        obj.diagnostics = diagnostics or {}
        return obj

    def __getitem__(self, item):
        if isinstance(item, str) and item == "diagnostics":
            return self.diagnostics
        return super().__getitem__(item)

class C2PAExifFastGuard:
    """
    Instant Pre-Flight C2PA & EXIF Provenance Guard (v2.0)
    Powered by MetadataGuard with granular byte-offset inspection and approved EXIF registry.
    Detects full and partial AI provenance markers (e.g., truncated JUMBF blocks),
    validates legitimate scanner/camera signatures, and exposes spoofed metadata.
    """
    def __init__(self):
        self.byte_parser = JUMBFByteParser()
        self.registry = ApprovedExifRegistry()
        self.metadata_guard = MetadataGuard(parser=self.byte_parser, registry=self.registry)

        self.c2pa_signatures = [
            b"trainedalgorithmicmedia",
            b"compositewithtrainedalgorithmicmedia",
            b"c2pa",
            b"jumbf",
            b"content credentials"
        ]
        self.ai_software_signatures = [
            "dall-e", "dalle", "chatgpt", "openai", "midjourney",
            "stable diffusion", "stablediffusion", "comfyui", "automatic1111",
            "novelai", "adobe firefly", "firefly", "flux.1", "black forest labs"
        ]

    def inspect_raw_c2pa(self, file_bytes: bytes) -> Tuple[bool, Optional[str]]:
        """
        Scans raw file header and JUMBF/XMP boxes with exact byte offset parsing.
        """
        parsed = self.byte_parser.parse(file_bytes)
        if parsed.has_ai_tags or parsed.truncated:
            for d in parsed.details:
                if d.tag_type == "TRUNCATED_JUMBF_BLOCK":
                    return True, f"Truncated C2PA JUMBF block at byte offset {d.offset_hex}: {d.context}"
                elif d.tag_type == "PARTIAL_AI_TAG":
                    return True, f"Partial/severed AI provenance tag at byte offset {d.offset_hex}: '{d.raw_snippet}'"
                elif d.tag_type in ["C2PA_MANIFEST", "IPTC_DIGITAL_SOURCE_TYPE"]:
                    return True, f"C2PA Content Credential at byte offset {d.offset_hex}: {d.context}"
                elif d.tag_type == "AI_PROVENANCE_SIGNATURE":
                    return True, f"AI generation marker at byte offset {d.offset_hex}: {d.context}"

        return False, None

    def inspect_exif_and_xmp(self, image_path: str) -> Tuple[bool, Optional[str]]:
        """
        Inspects EXIF, XMP, and format metadata chunks against approved whitelist registry.
        """
        eval_res = self.metadata_guard.evaluate(image_path)
        whitelist = eval_res.get("whitelist_status")

        # Check for spoofing
        if whitelist and whitelist.get("is_spoofed_attempt"):
            return True, f"Spoofed EXIF hardware tag detected: {whitelist.get('spoof_reason')}"

        # If flagged as AI or tampered by metadata guard
        if eval_res.get("is_ai") or eval_res.get("is_tampered_truncated"):
            for d in eval_res.get("byte_offset_diagnostics", []):
                if d.get("tag_type") == "SPOOFED_SCANNER_METADATA":
                    return True, f"Anti-forensic scanner spoofing detected: {d.get('context')}"
                elif d.get("severity") in ["CRITICAL", "HIGH"]:
                    return True, f"Metadata Alert [{d.get('tag_type')} at {d.get('offset_hex')}]: {d.get('context')}"

        # Filename fallback check
        filename = os.path.basename(image_path).lower()
        for kw in self.ai_software_signatures:
            if kw in filename:
                return True, f"Document filename contains AI generation tag: '{kw}'"

        return False, None

    def evaluate_fast_path(self, image_path: str) -> FastPathResult:
        """
        Fast-Path Check:
        Returns backward-compatible FastPathResult (is_ai, reason_title, reason_explanation)
        with extended diagnostics in .diagnostics.
        """
        eval_res = self.metadata_guard.evaluate(image_path)

        # 1. Truncated JUMBF / AI Provenance check
        if eval_res["is_tampered_truncated"]:
            trunc_items = [d for d in eval_res["byte_offset_diagnostics"] if "TRUNCATED" in d["tag_type"] or d["tag_type"] == "PARTIAL_AI_TAG"]
            first_msg = trunc_items[0]["context"] if trunc_items else "Truncated C2PA metadata block detected"
            first_off = trunc_items[0]["offset_hex"] if trunc_items else "0x00"
            return FastPathResult(
                True,
                "C2PA / JUMBF Truncation Alert (Tampered Synthetic Media)",
                f"Byte offset {first_off}: {first_msg}",
                diagnostics=eval_res
            )

        # 2. Confirmed AI Provenance Tag check
        if eval_res["is_ai"]:
            ai_items = [d for d in eval_res["byte_offset_diagnostics"] if d["tag_type"] in ["C2PA_MANIFEST", "IPTC_DIGITAL_SOURCE_TYPE", "AI_PROVENANCE_SIGNATURE", "AI_PROMPT_PARAMETER", "SPOOFED_SCANNER_METADATA"]]
            first_msg = ai_items[0]["context"] if ai_items else "Confirmed AI synthesis provenance"
            first_off = ai_items[0]["offset_hex"] if ai_items else "0x00"
            first_type = ai_items[0]["tag_type"] if ai_items else "AI_PROVENANCE"

            title = "C2PA Content Credentials (AI-Generated)" if "C2PA" in first_type or "IPTC" in first_type else "EXIF / Metadata Provenance (AI-Generated)"
            if first_type == "SPOOFED_SCANNER_METADATA":
                title = "Spoofed Scanner EXIF Metadata Alert"

            return FastPathResult(
                True,
                title,
                f"Byte offset {first_off}: {first_msg}",
                diagnostics=eval_res
            )

        # 3. Filename check for explicit AI tags
        filename = os.path.basename(image_path).lower()
        for kw in self.ai_software_signatures:
            if kw in filename:
                return FastPathResult(
                    True,
                    "EXIF / Metadata Provenance (AI-Generated)",
                    f"Document filename contains AI generation tag: '{kw}'",
                    diagnostics=eval_res
                )

        return FastPathResult(False, None, None, diagnostics=eval_res)

if __name__ == "__main__":
    guard = C2PAExifFastGuard()
    sample = r"E:\fraud Invoice&Receipt Detector\data\uploads\ChatGPT Image Sep 13, 2026, 06_00_24 PM.png"
    if os.path.exists(sample):
        res = guard.evaluate_fast_path(sample)
        print("Fast-Path Result:")
        print("Is AI:", res[0])
        print("Title:", res[1])
        print("Explanation:", res[2])
        print("Diagnostics:", res.diagnostics)
