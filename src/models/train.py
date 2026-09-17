import os
import glob
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from src.models.classifier import InvoiceFraudClassifier

class UserInvoiceDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

def train_user_datasets(epochs=4, batch_size=8, lr=1e-4, save_path=r"E:\fraud Invoice&Receipt Detector\models\resnet18_fraud.pth"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    real_receipts = glob.glob(r"D:\SOIRE\ICDAR-2019-SROIE-master\data\img\*.jpg")
    real_invoices = glob.glob(r"D:\Receipt-Invoice-fraud-detection\data\real\invoice\*.jpg")
    fake_receipts = glob.glob(r"D:\Receipt-Invoice-fraud-detection\data\fake\receipt\*.jpg")
    fake_invoices = glob.glob(r"D:\Receipt-Invoice-fraud-detection\data\fake\invoice\*.jpg")

    random.seed(42)
    random.shuffle(real_receipts)
    random.shuffle(real_invoices)
    random.shuffle(fake_receipts)
    random.shuffle(fake_invoices)

    selected_real = real_receipts[:50] + real_invoices[:50] # 100 real
    selected_fake = fake_receipts[:50] + fake_invoices[:50] # 100 fake
    samples = [(p, 0) for p in selected_real] + [(p, 1) for p in selected_fake]
    random.shuffle(samples)

    split = int(0.8 * len(samples))
    train_samples = samples[:split]
    val_samples = samples[split:]

    transform = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    train_loader = DataLoader(UserInvoiceDataset(train_samples, transform), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(UserInvoiceDataset(val_samples, transform), batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = InvoiceFraudClassifier(pretrained=True).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, lbls)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            _, preds = torch.max(out, 1)
            correct += (preds == lbls).sum().item()
            total += lbls.size(0)

        train_acc = (correct / (total or 1)) * 100.0
        train_loss = running_loss / (total or 1)

        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                out = model(imgs)
                _, preds = torch.max(out, 1)
                val_correct += (preds == lbls).sum().item()
                val_total += lbls.size(0)

        val_acc = (val_correct / (val_total or 1)) * 100.0
        print(f"ResNet-18 Epoch [{epoch}/{epochs}] - Loss: {train_loss:.4f} | Train Acc: {train_acc:.1f}% | Val Acc: {val_acc:.1f}%")

        if val_acc >= best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_path)

    print(f"ResNet-18 trained and saved to {save_path} (Best Val Acc: {best_acc:.1f}%)")

if __name__ == "__main__":
    train_user_datasets(epochs=3)
