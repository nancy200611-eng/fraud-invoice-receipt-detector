import os
import io
import cv2
import numpy as np
from PIL import Image, ImageChops

class DocumentForensics:
    """
    Enhanced Document Forensics Engine based on Thornton et al. (2025) & FUSION++ (BMVC 2024).
    Implements in-memory ultra-fast computation:
    - Bayar Constrained Convolution
    - Steganalysis Rich Models (SRM) High-Pass Filter Bank
    - Histogram of Oriented Gradients (HOG) Residuals
    - In-Memory Error Level Analysis (ELA)
    """
    def __init__(self, ela_quality=90, ela_scale=15):
        self.ela_quality = ela_quality
        self.ela_scale = ela_scale

        # 1. Bayar Constrained 5x5 Filter
        self.bayar_kernel = np.array([
            [-1,  2, -2,  2, -1],
            [ 2, -6,  8, -6,  2],
            [-2,  8, -1,  8, -2],
            [ 2, -6,  8, -6,  2],
            [-1,  2, -2,  2, -1]
        ], dtype=np.float32)
        mask_outer = np.ones((5, 5), dtype=bool)
        mask_outer[2, 2] = False
        self.bayar_kernel[mask_outer] /= np.sum(np.abs(self.bayar_kernel[mask_outer]))
        self.bayar_kernel[2, 2] = -1.0

        # 2. SRM Filter Bank
        self.srm_k1 = np.array([[0,  0, 0],
                                [-1, 2, -1],
                                [0,  0, 0]], dtype=np.float32)
        self.srm_k2 = np.array([[-1,  2, -1],
                                [ 2, -4,  2],
                                [-1,  2, -1]], dtype=np.float32)
        self.srm_k3 = np.array([[-1, -1, -1],
                                [-1,  8, -1],
                                [-1, -1, -1]], dtype=np.float32)

    def compute_bayar_residual(self, image_bgr):
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        bayar_resp = cv2.filter2D(gray, -1, self.bayar_kernel)
        bayar_abs = np.abs(bayar_resp)
        bayar_norm = cv2.normalize(bayar_abs, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        mean_resp = np.mean(bayar_abs)
        std_resp = np.std(bayar_abs)
        bayar_anomaly = float(std_resp / (mean_resp + 1e-5))

        return {
            "bayar_map": bayar_norm,
            "bayar_anomaly": round(bayar_anomaly, 3)
        }

    def compute_srm_residuals(self, image_bgr):
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        r1 = np.abs(cv2.filter2D(gray, -1, self.srm_k1))
        r2 = np.abs(cv2.filter2D(gray, -1, self.srm_k2))
        r3 = np.abs(cv2.filter2D(gray, -1, self.srm_k3))

        srm_comp = (r1 + r2 + r3) / 3.0
        srm_norm = cv2.normalize(srm_comp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        srm_variance = float(np.var(srm_comp))

        return {
            "srm_map": srm_norm,
            "srm_variance": round(srm_variance, 2)
        }

    def compute_hog_residuals(self, image_bgr):
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag, _ = cv2.cartToPolar(gx, gy, angleInDegrees=True)
        hog_energy = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        kernel = np.ones((9, 9), np.float32) / 81.0
        local_mag = cv2.filter2D(mag, -1, kernel)
        hog_anomaly = float(np.max(local_mag) / (np.mean(local_mag) + 1e-5))

        return {
            "hog_map": hog_energy,
            "hog_anomaly": round(hog_anomaly, 2)
        }

    def compute_ela(self, image_input):
        """
        Ultra-fast in-memory ELA without disk writes.
        image_input can be filepath string, PIL Image, or numpy BGR array.
        """
        if isinstance(image_input, str):
            orig = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, np.ndarray):
            orig = Image.fromarray(cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB))
        else:
            orig = image_input.convert("RGB")

        buf = io.BytesIO()
        orig.save(buf, "JPEG", quality=self.ela_quality)
        buf.seek(0)
        recompressed = Image.open(buf).convert("RGB")

        diff = ImageChops.difference(orig, recompressed)
        diff_np = np.array(diff, dtype=np.float32)
        ela_scaled = np.clip(diff_np * self.ela_scale, 0, 255).astype(np.uint8)
        mean_err = float(np.mean(diff_np))
        high_err_ratio = float(np.sum(diff_np > 14) / diff_np.size)

        return {
            "ela_map": ela_scaled,
            "mean_error": round(mean_err, 4),
            "high_error_ratio": round(high_err_ratio, 4)
        }

    def analyze_document(self, image_path):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        img_bgr = cv2.imread(image_path)
        bayar_res = self.compute_bayar_residual(img_bgr)
        srm_res = self.compute_srm_residuals(img_bgr)
        hog_res = self.compute_hog_residuals(img_bgr)
        ela_res = self.compute_ela(img_bgr)

        bayar_g = bayar_res["bayar_map"].astype(np.float32)
        srm_g = srm_res["srm_map"].astype(np.float32)
        hog_g = hog_res["hog_map"].astype(np.float32)
        ela_g = cv2.cvtColor(ela_res["ela_map"], cv2.COLOR_RGB2GRAY).astype(np.float32)

        fused_energy = (bayar_g * 0.35 + srm_g * 0.30 + hog_g * 0.20 + ela_g * 0.15)
        smooth_energy = cv2.GaussianBlur(fused_energy, (19, 19), 0)
        norm_energy = cv2.normalize(smooth_energy, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        heatmap_color = cv2.applyColorMap(norm_energy, cv2.COLORMAP_JET)
        blended_overlay = cv2.addWeighted(heatmap_color, 0.45, img_bgr, 0.55, 0)

        thresh_val = max(160, int(np.mean(norm_energy) + 2.2 * np.std(norm_energy)))
        _, raw_mask = cv2.threshold(norm_energy, min(235, thresh_val), 255, cv2.THRESH_BINARY)
        
        kernel_enh = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
        enhanced_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, kernel_enh)
        enhanced_mask = cv2.morphologyEx(enhanced_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))

        forensic_score = min(1.0, (
            (bayar_res["bayar_anomaly"] / 6.0) * 0.35 +
            (ela_res["high_error_ratio"] * 3.0) * 0.30 +
            (hog_res["hog_anomaly"] / 15.0) * 0.20 +
            (min(srm_res["srm_variance"], 500) / 500.0) * 0.15
        ))

        return {
            "forensic_score": round(forensic_score, 4),
            "bayar_metrics": bayar_res,
            "srm_metrics": srm_res,
            "hog_metrics": hog_res,
            "ela_metrics": ela_res,
            "heatmap_image": blended_overlay,
            "predicted_mask": raw_mask,
            "enhanced_mask": enhanced_mask
        }

if __name__ == "__main__":
    sample = r"D:\SOIRE\ICDAR-2019-SROIE-master\data\img\000.jpg"
    forensics = DocumentForensics()
    res = forensics.analyze_document(sample)
    print("In-Memory ELA & Forensics OK! Score:", res["forensic_score"])
