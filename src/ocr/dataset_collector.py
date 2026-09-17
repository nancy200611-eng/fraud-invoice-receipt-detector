import os
import random
from datetime import datetime, timedelta
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2

VENDORS = [
    ("APEX RETAIL SOLUTIONS", "1024 Market Blvd, Austin, TX", "(512) 555-0199"),
    ("METRO GROCERY MART", "459 High Street, Seattle, WA", "(206) 555-4820"),
    ("COSMO HARDWARE SUPPLIES", "88 Industrial Way, Chicago, IL", "(312) 555-7714"),
    ("ZENITH OFFICE TECH", "300 Silicon Ave, San Jose, CA", "(408) 555-9321"),
    ("SUNRISE CAFE & BISTRO", "12 Harbour Rd, Boston, MA", "(617) 555-6632")
]

ITEMS = [
    ("Wireless Mouse M220", 24.99),
    ("USB-C Fast Cable 2M", 12.50),
    ("A4 Copy Paper 500s", 8.99),
    ("Coffee Beans 1kg Roast", 18.00),
    ("Mechanical Keyboard RGB", 79.99),
    ("Desk LED Lamp Dim", 34.50),
    ("Alkaline Battery AA 8pk", 9.99),
    ("Ballpoint Pens 12pk", 5.49),
    ("Ergonomic Wrist Rest", 15.00),
    ("Monitor Cleaning Kit", 11.25)
]

def create_synthetic_receipt(output_path, receipt_id=None):
    """
    Generates a realistic synthetic thermal/standard receipt image
    with known structured ground truth fields.
    """
    width = 600
    height = 850
    # Slight off-white / parchment background
    bg_color = (random.randint(245, 252), random.randint(245, 252), random.randint(240, 248))
    img = Image.new("RGB", (width, height), color=bg_color)
    draw = ImageDraw.Draw(img)

    # Use default font or fallback
    try:
        font_large = ImageFont.truetype("arial.ttf", 24)
        font_med = ImageFont.truetype("arial.ttf", 16)
        font_small = ImageFont.truetype("cour.ttf", 15)
        font_bold = ImageFont.truetype("arialbd.ttf", 17)
    except Exception:
        font_large = ImageFont.load_default()
        font_med = ImageFont.load_default()
        font_small = ImageFont.load_default()
        font_bold = ImageFont.load_default()

    vendor, address, phone = random.choice(VENDORS)
    if receipt_id is None:
        receipt_id = f"INV-{random.randint(10000, 99999)}"
    
    date_obj = datetime.now() - timedelta(days=random.randint(1, 90))
    date_str = date_obj.strftime("%Y-%m-%d %H:%M")

    # Header
    y = 40
    draw.text((width // 2, y), vendor, fill=(20, 20, 20), font=font_large, anchor="mm")
    y += 30
    draw.text((width // 2, y), address, fill=(60, 60, 60), font=font_med, anchor="mm")
    y += 22
    draw.text((width // 2, y), f"Tel: {phone}", fill=(60, 60, 60), font=font_med, anchor="mm")
    y += 30

    # Separator line
    draw.line([(40, y), (width - 40, y)], fill=(120, 120, 120), width=1)
    y += 18

    # Metadata
    draw.text((50, y), f"INVOICE #: {receipt_id}", fill=(30, 30, 30), font=font_bold)
    y += 24
    draw.text((50, y), f"DATE: {date_str}", fill=(50, 50, 50), font=font_small)
    draw.text((width - 50, y), "CASHIER: #04", fill=(50, 50, 50), font=font_small, anchor="ra")
    y += 30

    # Table Header
    draw.line([(40, y), (width - 40, y)], fill=(160, 160, 160), width=1)
    y += 10
    draw.text((50, y), "ITEM DESCRIPTION", fill=(30, 30, 30), font=font_bold)
    draw.text((370, y), "QTY", fill=(30, 30, 30), font=font_bold)
    draw.text((width - 50, y), "AMOUNT", fill=(30, 30, 30), font=font_bold, anchor="ra")
    y += 25
    draw.line([(40, y), (width - 40, y)], fill=(160, 160, 160), width=1)
    y += 15

    # Line Items
    num_items = random.randint(3, 5)
    selected_items = random.sample(ITEMS, num_items)
    subtotal = 0.0
    field_metadata = {
        "vendor": vendor,
        "invoice_id": receipt_id,
        "date": date_str,
        "items": []
    }

    for name, price in selected_items:
        qty = random.randint(1, 3)
        item_total = round(qty * price, 2)
        subtotal += item_total
        
        draw.text((50, y), name, fill=(35, 35, 35), font=font_small)
        draw.text((380, y), str(qty), fill=(35, 35, 35), font=font_small)
        draw.text((width - 50, y), f"${item_total:.2f}", fill=(35, 35, 35), font=font_small, anchor="ra")
        
        field_metadata["items"].append({
            "name": name,
            "qty": qty,
            "price": price,
            "total": item_total,
            "y_pos": y
        })
        y += 28

    subtotal = round(subtotal, 2)
    tax = round(subtotal * 0.0825, 2) # 8.25% tax
    grand_total = round(subtotal + tax, 2)

    field_metadata["subtotal"] = subtotal
    field_metadata["tax"] = tax
    field_metadata["total"] = grand_total

    y += 15
    draw.line([(40, y), (width - 40, y)], fill=(120, 120, 120), width=1)
    y += 20

    # Totals
    draw.text((320, y), "SUBTOTAL:", fill=(40, 40, 40), font=font_small)
    draw.text((width - 50, y), f"${subtotal:.2f}", fill=(40, 40, 40), font=font_small, anchor="ra")
    y += 26
    draw.text((320, y), "TAX (8.25%):", fill=(40, 40, 40), font=font_small)
    draw.text((width - 50, y), f"${tax:.2f}", fill=(40, 40, 40), font=font_small, anchor="ra")
    y += 30

    # Grand Total (Prominent)
    draw.line([(300, y), (width - 40, y)], fill=(80, 80, 80), width=2)
    y += 12
    draw.text((280, y), "TOTAL DUE:", fill=(10, 10, 10), font=font_bold)
    total_bbox = (width - 160, y - 2, width - 40, y + 26)
    draw.text((width - 50, y), f"${grand_total:.2f}", fill=(10, 10, 10), font=font_large, anchor="ra")
    field_metadata["total_box"] = total_bbox
    y += 45

    # Footer
    draw.line([(40, y), (width - 40, y)], fill=(180, 180, 180), width=1)
    y += 20
    draw.text((width // 2, y), "THANK YOU FOR YOUR BUSINESS!", fill=(70, 70, 70), font=font_med, anchor="mm")
    y += 24
    draw.text((width // 2, y), "Please retain receipt for returns within 30 days", fill=(100, 100, 100), font=font_small, anchor="mm")

    # Add realistic paper degradation (noise, subtle gradient/shadow, slight blur)
    img_np = np.array(img)
    noise = np.random.normal(0, 3.5, img_np.shape).astype(np.int16)
    noisy_img = np.clip(img_np.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    # Subtle Gaussian smoothing to emulate scanner lens
    noisy_img = cv2.GaussianBlur(noisy_img, (3, 3), 0.3)

    final_pil = Image.fromarray(noisy_img)
    final_pil.save(output_path, "JPEG", quality=92)
    return field_metadata

def generate_sample_dataset(count=20, output_dir=r"E:\fraud Invoice&Receipt Detector\data\raw"):
    os.makedirs(output_dir, exist_ok=True)
    generated = []
    print(f"Generating {count} authentic synthetic invoices...")
    for i in range(count):
        filename = f"invoice_{i+1:04d}.jpg"
        filepath = os.path.join(output_dir, filename)
        meta = create_synthetic_receipt(filepath, receipt_id=f"INV-2026-{i+1:04d}")
        meta["filename"] = filename
        meta["filepath"] = filepath
        generated.append(meta)
    print(f"Successfully generated {len(generated)} genuine invoices in {output_dir}")
    return generated

if __name__ == "__main__":
    generate_sample_dataset(15)
