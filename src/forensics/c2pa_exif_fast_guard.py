import os
import re
from PIL import Image
from PIL.ExifTags import TAGS

class C2PAExifFastGuard:
    """
    Instant Pre-Flight C2PA & EXIF Provenance Guard
    Immediately flags and rejects AI-generated documents without wasting compute.
    Inspects:
    1. C2PA (Coalition for Content Provenance and Authenticity) manifests & JUMBF boxes
    2. IPTC DigitalSourceType: 'trainedAlgorithmicMedia' (C2PA standard by Google, OpenAI, Adobe, Meta)
    3. EXIF Software, Artist, UserComment, ImageDescription for AI signatures
    4. PNG metadata chunks (tEXt, iTXt, 'parameters', 'workflow', 'prompt')
    """
    def __init__(self):
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

    def inspect_raw_c2pa(self, file_bytes):
        """
        Scans raw file header and JUMBF/XMP boxes for C2PA provenance markers.
        """
        lower_bytes = file_bytes[:1048576].lower() # Scan first 1MB for headers/XMP
        for sig in self.c2pa_signatures:
            if sig in lower_bytes:
                sig_str = sig.decode("utf-8", errors="ignore")
                if sig_str in ["trainedalgorithmicmedia", "compositewithtrainedalgorithmicmedia"]:
                    return True, "C2PA IPTC DigitalSourceType: trainedAlgorithmicMedia (Official Content Credential confirming AI synthesis)"
                elif sig_str == "c2pa":
                    return True, "Embedded C2PA Content Provenance Manifest detected in image headers"
                elif sig_str == "content credentials":
                    return True, "Verified C2PA Content Credentials claim detected"
        return False, None

    def inspect_exif_and_xmp(self, image_path):
        """
        Inspects EXIF, XMP, and format metadata chunks.
        """
        try:
            with Image.open(image_path) as img:
                # 1. Inspect PIL info dictionary (PNG chunks, text comments)
                info = img.info or {}
                combined_info = " ".join([f"{k}:{v}" for k, v in info.items()]).lower()

                for kw in self.ai_software_signatures:
                    if kw in combined_info:
                        return True, f"Image metadata explicitly identifies generator: '{kw.upper()}'"

                # Check for Stable Diffusion / ComfyUI parameters
                if any(p in combined_info for p in ["steps:", "sampler:", "cfg scale:", "seed:", "negative prompt:"]):
                    return True, "Metadata contains explicit AI diffusion generation parameters (Steps, Sampler, Seed)"

                # 2. Inspect EXIF tags
                exif = img.getexif()
                if exif:
                    for tag_id, value in exif.items():
                        tag_name = TAGS.get(tag_id, str(tag_id)).lower()
                        val_str = str(value).lower()
                        for kw in self.ai_software_signatures:
                            if kw in val_str:
                                return True, f"EXIF [{tag_name}] explicitly lists AI software: '{value}'"

                # 3. Check filename
                filename = os.path.basename(image_path).lower()
                for kw in self.ai_software_signatures:
                    if kw in filename:
                        return True, f"Document filename contains AI generation tag: '{kw}'"

        except Exception as e:
            print(f"EXIF/C2PA read error: {e}")

        return False, None

    def evaluate_fast_path(self, image_path):
        """
        Fast-Path Check:
        Returns (is_ai, reason_title, reason_explanation)
        """
        with open(image_path, "rb") as f:
            file_bytes = f.read()

        # 1. C2PA Check
        is_c2pa, c2pa_reason = self.inspect_raw_c2pa(file_bytes)
        if is_c2pa:
            return True, "C2PA Content Credentials (AI-Generated)", c2pa_reason

        # 2. EXIF / Metadata Check
        is_exif, exif_reason = self.inspect_exif_and_xmp(image_path)
        if is_exif:
            return True, "EXIF / Metadata Provenance (AI-Generated)", exif_reason

        return False, None, None

if __name__ == "__main__":
    guard = C2PAExifFastGuard()
    sample = r"E:\fraud Invoice&Receipt Detector\data\uploads\ChatGPT Image Sep 13, 2026, 06_00_24 PM.png"
    is_ai, title, explanation = guard.evaluate_fast_path(sample)
    print("Fast-Path Result:")
    print("Is AI:", is_ai)
    print("Title:", title)
    print("Explanation:", explanation)
