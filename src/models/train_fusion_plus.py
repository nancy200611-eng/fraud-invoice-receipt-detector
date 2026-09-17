import os
import glob
import random
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from src.models.fusion_plus import FusionPlusModel
from src.forensics.image_forensics import DocumentForensics

def build_and_train(epochs=5, batch_size=8, lr=3e-4, save_path=r"E:\fraud Invoice&Receipt Detector\models\fusion_plus_weights.pth"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    forensics = DocumentForensics()

    real_receipts = glob.glob(r"D:\SOIRE\ICDAR-2019-SROIE-master\data\img\*.jpg")
    real_invoices = glob.glob(r"D:\Receipt-Invoice-fraud-detection\data\real\invoice\*.jpg")
    fake_receipts = glob.glob(r"D:\Receipt-Invoice-fraud-detection\data\fake\receipt\*.jpg")
    fake_invoices = glob.glob(r"D:\Receipt-Invoice-fraud-detection\data\fake\invoice\*.jpg")

    random.seed(42)
    random.shuffle(real_receipts)
    random.shuffle(real_invoices)
    random.shuffle(fake_receipts)
    random.shuffle(fake_invoices)

    # 25 real receipts + 25 real invoices = 50 real
    # 25 fake receipts + 25 fake invoices = 50 fake
    selected_real = real_receipts[:25] + real_invoices[:25]
    selected_fake = fake_receipts[:25] + fake_invoices[:25]
    all_pairs = [(p, 0) for p in selected_real] + [(p, 1) for p in selected_fake]
    random.shuffle(all_pairs)

    print(f"Pre-extracting FUSION++ dual-path tensors for {len(all_pairs)} benchmark documents from D: drive...")
    rgb_list = []
    flt_list = []
    lbl_list = []
    target_size = (256, 256)

    for idx, (path, label) in enumerate(all_pairs):
        img_bgr = cv2.imread(path)
        if img_bgr is None:
            continue
        img_res = cv2.resize(img_bgr, target_size)
        rgb_norm = cv2.cvtColor(img_res, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        bayar = cv2.resize(forensics.compute_bayar_residual(img_bgr)["bayar_map"], target_size).astype(np.float32) / 255.0
        srm = cv2.resize(forensics.compute_srm_residuals(img_bgr)["srm_map"], target_size).astype(np.float32) / 255.0
        hog = cv2.resize(forensics.compute_hog_residuals(img_bgr)["hog_map"], target_size).astype(np.float32) / 255.0
        ela = cv2.resize(cv2.cvtColor(forensics.compute_ela(img_bgr)["ela_map"], cv2.COLOR_RGB2GRAY), target_size).astype(np.float32) / 255.0

        flt_stack = np.stack([bayar, srm, hog, ela], axis=0) # [4, 256, 256]
        rgb_stack = np.transpose(rgb_norm, (2, 0, 1))        # [3, 256, 256]

        rgb_list.append(rgb_stack)
        flt_list.append(flt_stack)
        lbl_list.append(label)

    rgb_tensor = torch.from_numpy(np.array(rgb_list)).float()
    flt_tensor = torch.from_numpy(np.array(flt_list)).float()
    lbl_tensor = torch.tensor(lbl_list, dtype=torch.long)

    total_count = len(lbl_list)
    split = int(0.8 * total_count)

    train_ds = TensorDataset(rgb_tensor[:split], flt_tensor[:split], lbl_tensor[:split])
    val_ds = TensorDataset(rgb_tensor[split:], flt_tensor[split:], lbl_tensor[split:])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    print(f"Extracted {total_count} pairs ({split} train, {total_count - split} val). Training FUSION++...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FusionPlusModel().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for rgb_b, flt_b, y_b in train_loader:
            rgb_b, flt_b, y_b = rgb_b.to(device), flt_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            logits, mask = model(rgb_b, flt_b)
            loss = criterion(logits, y_b)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * rgb_b.size(0)
            _, preds = torch.max(logits, 1)
            correct += (preds == y_b).sum().item()
            total += y_b.size(0)

        train_acc = (correct / (total or 1)) * 100.0
        train_loss = running_loss / (total or 1)

        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for rgb_b, flt_b, y_b in val_loader:
                rgb_b, flt_b, y_b = rgb_b.to(device), flt_b.to(device), y_b.to(device)
                logits, mask = model(rgb_b, flt_b)
                _, preds = torch.max(logits, 1)
                val_correct += (preds == y_b).sum().item()
                val_total += y_b.size(0)

        val_acc = (val_correct / (val_total or 1)) * 100.0
        print(f"Epoch [{epoch}/{epochs}] - Loss: {train_loss:.4f} | Train Acc: {train_acc:.1f}% | Val Acc: {val_acc:.1f}%")

        if val_acc >= best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_path)

    print(f"FUSION++ training complete! Model saved to: {save_path} (Best Val Acc: {best_acc:.1f}%)")
    return save_path

if __name__ == "__main__":
    build_and_train(epochs=5)
