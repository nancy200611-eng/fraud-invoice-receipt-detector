import os
import random
import glob
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

class InvoiceManipulator:
    """
    Phase 2: Gravity-Defying Fakes Manipulation Engine
    Generates realistic manipulations on invoices:
    - Text replacement & inpainting (altering Total, Subtotal, Invoice ID)
    - Font/compression mismatch simulation
    - Ground truth binary mask generation (0: clean, 255: manipulated)
    """
    def __init__(self):
        try:
            self.font_bold = ImageFont.truetype("arialbd.ttf", 26)
            self.font_alt = ImageFont.truetype("times.ttf", 25)
            self.font_mono = ImageFont.truetype("cour.ttf", 24)
        except Exception:
            self.font_bold = ImageFont.load_default()
            self.font_alt = ImageFont.load_default()
            self.font_mono = ImageFont.load_default()

    def manipulate_total_amount(self, image_path, output_img_path, output_mask_path):
        """
        Locates or targets the Total Due amount on the invoice,
        inpaints/erases the original number, overlays a higher forged amount ($50 -> $500),
        and adds realistic compression/edge artifacts.
        """
        img_bgr = cv2.imread(image_path)
        h, w = img_bgr.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        # Standard position for Total Due in synthetic/receipt layouts:
        # Lower right quadrant: x in [w-220, w-40], y in [h-260, h-180]
        # We find non-white pixels in that region to detect exact text boundary
        roi_y1, roi_y2 = int(h * 0.65), int(h * 0.85)
        roi_x1, roi_x2 = int(w * 0.55), int(w * 0.95)

        sub_roi = img_bgr[roi_y1:roi_y2, roi_x1:roi_x2]
        gray_sub = cv2.cvtColor(sub_roi, cv2.COLOR_BGR2GRAY)
        
        # Dark text threshold
        _, text_thresh = cv2.threshold(gray_sub, 120, 255, cv2.THRESH_BINARY_INV)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 4))
        dilated = cv2.dilate(text_thresh, kernel, iterations=1)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        target_box = None
        # Find the largest/most prominent contour in the total area
        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            if cw > 40 and ch > 14 and (x + cw) > (roi_x2 - roi_x1) * 0.5:
                target_box = (roi_x1 + x - 6, roi_y1 + y - 4, roi_x1 + x + cw + 8, roi_y1 + y + ch + 6)
                break

        if target_box is None:
            # Fallback default total bounding box
            target_box = (w - 200, int(h * 0.72), w - 45, int(h * 0.77))

        bx1, by1, bx2, by2 = target_box
        bx1, by1 = max(0, bx1), max(0, by1)
        bx2, by2 = min(w, bx2), min(h, by2)

        # Inpainting: remove original total
        inpaint_mask = np.zeros((h, w), dtype=np.uint8)
        inpaint_mask[by1:by2, bx1:bx2] = 255
        
        # Inpaint with Telea algorithm
        inpainted_bgr = cv2.inpaint(img_bgr, inpaint_mask, 5, cv2.INPAINT_TELEA)

        # Generate fraudulent number (e.g. inflating 3x-10x)
        fake_amount = random.choice([
            f"${random.randint(450, 990)}.{random.randint(10, 99):02d}",
            f"${random.randint(1200, 3500)}.{random.randint(10, 99):02d}"
        ])

        # Convert to PIL for anti-aliased font rendering
        pil_img = Image.fromarray(cv2.cvtColor(inpainted_bgr, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)

        # Deliberately pick slightly inconsistent font/color to mimic forgery
        font = random.choice([self.font_bold, self.font_alt, self.font_mono])
        text_color = (random.randint(5, 30), random.randint(5, 30), random.randint(5, 30))
        
        # Draw new inflated value
        draw.text((bx2 - 5, by1 + 4), fake_amount, fill=text_color, font=font, anchor="ra")

        manipulated_rgb = np.array(pil_img)
        manipulated_bgr = cv2.cvtColor(manipulated_rgb, cv2.COLOR_RGB2BGR)

        # Mark ground truth binary mask
        mask[by1:by2, bx1:bx2] = 255

        # Add splicing artifact: simulate local JPEG recompression mismatch
        # Save patch at lower JPEG quality and re-paste
        patch = manipulated_bgr[by1:by2, bx1:bx2]
        if patch.size > 0:
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), random.randint(60, 75)]
            _, encimg = cv2.imencode('.jpg', patch, encode_param)
            compressed_patch = cv2.imdecode(encimg, 1)
            manipulated_bgr[by1:by2, bx1:bx2] = compressed_patch

        # Save manipulated image and mask
        os.makedirs(os.path.dirname(output_img_path), exist_ok=True)
        os.makedirs(os.path.dirname(output_mask_path), exist_ok=True)
        
        # Save overall image with standard quality
        cv2.imwrite(output_img_path, manipulated_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        cv2.imwrite(output_mask_path, mask)

        return {
            "fake_amount": fake_amount,
            "manipulation_type": "total_inflation",
            "box": [bx1, by1, bx2, by2]
        }

    def generate_paired_dataset(self, raw_dir, out_images_dir, out_masks_dir, max_samples=20):
        os.makedirs(out_images_dir, exist_ok=True)
        os.makedirs(out_masks_dir, exist_ok=True)
        
        raw_images = glob.glob(os.path.join(raw_dir, "*.jpg")) + glob.glob(os.path.join(raw_dir, "*.png"))
        print(f"Generating manipulated fakes from {len(raw_images)} raw images...")
        
        manifest = []
        for idx, img_path in enumerate(raw_images[:max_samples]):
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            out_img = os.path.join(out_images_dir, f"fake_{base_name}.jpg")
            out_mask = os.path.join(out_masks_dir, f"fake_{base_name}_mask.png")
            
            meta = self.manipulate_total_amount(img_path, out_img, out_mask)
            meta["original_image"] = img_path
            meta["manipulated_image"] = out_img
            meta["mask_path"] = out_mask
            manifest.append(meta)

        print(f"Generated {len(manifest)} paired fake invoices & ground-truth masks.")
        return manifest

if __name__ == "__main__":
    raw_dir = r"E:\fraud Invoice&Receipt Detector\data\raw"
    out_img = r"E:\fraud Invoice&Receipt Detector\data\manipulated\images"
    out_msk = r"E:\fraud Invoice&Receipt Detector\data\manipulated\masks"
    
    manipulator = InvoiceManipulator()
    manipulator.generate_paired_dataset(raw_dir, out_img, out_msk)
