import os
import re
import cv2
import numpy as np
from PIL import Image
from PIL.ExifTags import TAGS
from src.forensics.approved_exif_registry import ApprovedExifRegistry

class AIGeneratorDetector:
    """
    Universal Synthetic & Generative AI Document Detector.
    Detects documents created by:
    - OpenAI ChatGPT & DALL-E (2 & 3)
    - Midjourney (v4, v5, v6)
    - Stable Diffusion (SD 1.5, SD 2.1, SDXL, Turbo) & ComfyUI / Automatic1111
    - Flux.1 (Schnell, Dev, Pro)
    - Adobe Firefly & Canva AI
    - TextDiffuser-2 & Diffusion inpainting models
    - Online digital receipt/invoice generators (ExpressExpense, InvoiceMaker, etc.)
    """
    def __init__(self):
        # Universal AI generator resolutions across Stable Diffusion, Flux, Midjourney, DALL-E
        self.ai_resolutions = {
            (512, 512), (768, 768), (1024, 1024), (1024, 1536), (1536, 1024),
            (1024, 1792), (1792, 1024), (896, 1152), (1152, 896), (832, 1216),
            (1216, 832), (1024, 768), (768, 1024), (1456, 816), (816, 1456)
        }
        
        # Engine keywords across all major models & tools
        self.ai_engine_signatures = {
            "ChatGPT / DALL-E": ["chatgpt", "dall-e", "dalle", "openai"],
            "Midjourney": ["midjourney"],
            "Stable Diffusion / SDXL": ["stable diffusion", "stablediffusion", "sdxl", "comfyui", "automatic1111", "a1111", "webui"],
            "Flux": ["flux", "bfl", "black forest labs"],
            "Adobe Firefly / Canva": ["firefly", "canva", "adobe sensei"],
            "TextDiffuser / Inpainting": ["textdiffuser", "anytext", "inpaint", "deepfloyd"],
            "Online Receipt Builder": ["expressexpense", "invoicemaker", "samourai", "receiptmaker", "fakereceipt"]
        }
        self.approved_registry = ApprovedExifRegistry()

    def inspect_metadata_and_signatures(self, image_path):
        detected_engines = []
        reasons = []
        ai_score = 0.0
        filename = os.path.basename(image_path).lower()

        # 1. Inspect Filename for known generator patterns
        for engine, keywords in self.ai_engine_signatures.items():
            for kw in keywords:
                if kw in filename:
                    detected_engines.append(engine)
                    ai_score += 0.85
                    reasons.append(f"Filename signature matches {engine} export ('{kw}')")
                    break

        # 2. Inspect Internal Metadata Chunks (PNG tEXt, iTXt, EXIF UserComment, Software)
        try:
            with Image.open(image_path) as img:
                w, h = img.size
                img_format = img.format
                img_mode = img.mode
                info = img.info or {}

                # Check text metadata chunks (Stable Diffusion / ComfyUI / Midjourney store prompt parameters)
                combined_info = " ".join([f"{k}:{v}" for k, v in info.items()]).lower()
                for engine, keywords in self.ai_engine_signatures.items():
                    for kw in keywords:
                        if kw in combined_info:
                            if engine not in detected_engines:
                                detected_engines.append(engine)
                            ai_score += 0.90
                            reasons.append(f"Internal file metadata contains {engine} generator trace: '{kw}'")
                            break

                # Detect Stable Diffusion / ComfyUI prompt parameter chunks
                if any(p in combined_info for p in ["steps:", "sampler:", "cfg scale:", "seed:", "prompt:"]):
                    if "Stable Diffusion / SDXL" not in detected_engines:
                        detected_engines.append("Stable Diffusion / SDXL")
                    ai_score += 0.95
                    reasons.append("Contains AI diffusion prompt parameters (Steps, Sampler, Seed, CFG scale)")

                # 3. Canvas Dimensions Check across all models verified with Approved Registry
                exif = img.getexif()
                exif_dict = {}
                if exif:
                    for tag_id, val in exif.items():
                        tag_name = TAGS.get(tag_id, str(tag_id))
                        if tag_name in ["Make", "Model", "Software"]:
                            exif_dict[tag_name] = str(val)

                whitelist_profile = self.approved_registry.evaluate_exif(exif_dict)
                has_real_camera = (whitelist_profile is not None and not whitelist_profile.is_spoofed_attempt)

                if whitelist_profile and whitelist_profile.is_spoofed_attempt:
                    ai_score += 0.80
                    reasons.append(f"Spoofed camera metadata detected: {whitelist_profile.spoof_reason}")

                if (w, h) in self.ai_resolutions and not has_real_camera:
                    ai_score += 0.65
                    reasons.append(f"Fixed AI latent canvas dimensions ({w}x{h}) matching Midjourney / Stable Diffusion / DALL-E without physical camera EXIF metadata")

                # 4. Check for Digital Vector / Template Flatness (Online Receipt Builders)
                # Online fake receipt tools produce zero camera sensor noise (perfect uniform background)
                gray = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2GRAY)
                blur = cv2.medianBlur(gray, 3)
                diff_noise = cv2.absdiff(gray, blur)
                noise_floor = float(np.mean(diff_noise))
                
                if noise_floor < 0.60 and not has_real_camera:
                    ai_score += 0.55
                    reasons.append(f"Unnaturally flat digital background (noise floor: {noise_floor:.2f}), typical of vector receipt builders or digital template generators")

        except Exception as e:
            print(f"Metadata read error: {e}")

        engine_str = ", ".join(set(detected_engines)) if detected_engines else "AI Diffusion Model"
        return ai_score, engine_str, reasons

    def inspect_glyph_deformities(self, tokens):
        """
        Detects universal diffusion text rendering artifacts (Midjourney, SDXL, Flux, DALL-E)
        """
        glyph_flags = []
        deformed_pattern = re.compile(r"^[?;{}\[\]!~@#^&*]+\s*\d+[\.,]\d{2}")
        low_conf_count = 0

        for t in tokens:
            text = t.get("text", "")
            conf = t.get("confidence", 1.0)
            if deformed_pattern.search(text):
                glyph_flags.append(f"AI glyph hallucination on amount token: '{text}'")
            if conf < 0.35 and len(text) > 3:
                low_conf_count += 1

        if low_conf_count >= 3:
            glyph_flags.append(f"Multiple deformed character glyphs ({low_conf_count} low-confidence tokens) caused by diffusion text rendering")

        return glyph_flags

    def detect_ai(self, image_path, tokens):
        score, engine_name, meta_reasons = self.inspect_metadata_and_signatures(image_path)
        glyph_reasons = self.inspect_glyph_deformities(tokens)
        all_reasons = meta_reasons + glyph_reasons

        if glyph_reasons:
            score += 0.35 * len(glyph_reasons)

        ai_confidence = round(min(100.0, score * 100.0), 1)
        is_ai = bool(ai_confidence >= 55.0 or len(meta_reasons) >= 1)

        return {
            "is_ai_generated": is_ai,
            "engine_detected": engine_name,
            "ai_confidence": ai_confidence,
            "reasons": all_reasons
        }

if __name__ == "__main__":
    det = AIGeneratorDetector()
    print("Universal AI Generator Detector loaded with signatures for:")
    for eng in det.ai_engine_signatures.keys():
        print(f" - {eng}")
