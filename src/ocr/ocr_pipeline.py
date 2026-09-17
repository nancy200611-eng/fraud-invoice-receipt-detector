import os
import glob
import csv
import cv2
import numpy as np

class ReceiptOCRPipeline:
    def __init__(self, use_gpu=False):
        self.use_gpu = use_gpu
        self.reader = None
        self._init_reader()

    def _init_reader(self):
        try:
            import easyocr
            self.reader = easyocr.Reader(['en'], gpu=self.use_gpu, verbose=False)
        except Exception as e:
            print(f"EasyOCR init warning: {e}")
            self.reader = None

    def extract_from_image(self, image_path):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = cv2.imread(image_path)
        h, w = image.shape[:2]
        results = []

        if self.reader is not None:
            try:
                raw_results = self.reader.readtext(image_path)
                for bbox, text, conf in raw_results:
                    pts = np.array(bbox).astype(int)
                    xmin = int(np.min(pts[:, 0]))
                    ymin = int(np.min(pts[:, 1]))
                    xmax = int(np.max(pts[:, 0]))
                    ymax = int(np.max(pts[:, 1]))
                    results.append({
                        "text": text.strip(),
                        "confidence": float(round(conf, 4)),
                        "box": [max(0, xmin), max(0, ymin), min(w, xmax), min(h, ymax)],
                        "polygon": pts.tolist()
                    })
                return results
            except Exception as e:
                print(f"EasyOCR error: {e}")

        # Fallback if reader fails
        return []

    def export_to_csv(self, image_paths, output_csv_path):
        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
        all_rows = []

        print(f"Running OCR extraction on {len(image_paths)} documents...")
        for img_path in image_paths:
            filename = os.path.basename(img_path)
            detections = self.extract_from_image(img_path)
            for det in detections:
                box = det["box"]
                all_rows.append({
                    "filename": filename,
                    "text": det["text"],
                    "confidence": det["confidence"],
                    "x_min": box[0],
                    "y_min": box[1],
                    "x_max": box[2],
                    "y_max": box[3],
                    "box_width": box[2] - box[0],
                    "box_height": box[3] - box[1]
                })

        with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "filename", "text", "confidence", "x_min", "y_min", "x_max", "y_max", "box_width", "box_height"
            ])
            writer.writeheader()
            writer.writerows(all_rows)

        print(f"Exported {len(all_rows)} text token records to {output_csv_path}")
        return output_csv_path

    def draw_detections(self, image_path, detections, output_path):
        image = cv2.imread(image_path)
        overlay = image.copy()
        for det in detections:
            box = det["box"]
            cv2.rectangle(overlay, (box[0], box[1]), (box[2], box[3]), (0, 210, 80), -1)
            cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 180, 0), 2)
            label = f"{det['text'][:15]} ({int(det['confidence']*100)}%)"
            cv2.putText(image, label, (box[0], max(15, box[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (20, 20, 180), 1)

        cv2.addWeighted(overlay, 0.25, image, 0.75, 0, image)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, image)
        return output_path

if __name__ == "__main__":
    raw_dir = r"E:\fraud Invoice&Receipt Detector\data\raw"
    images = glob.glob(os.path.join(raw_dir, "*.jpg"))[:10]
    
    pipeline = ReceiptOCRPipeline(use_gpu=False)
    csv_out = r"E:\fraud Invoice&Receipt Detector\data\ocr_extracted\extracted_tokens.csv"
    pipeline.export_to_csv(images, csv_out)

    if images:
        dets = pipeline.extract_from_image(images[0])
        vis_out = r"E:\fraud Invoice&Receipt Detector\data\ocr_extracted\sample_ocr_visualized.jpg"
        pipeline.draw_detections(images[0], dets, vis_out)
        print(f"Visualized OCR saved to {vis_out}")
