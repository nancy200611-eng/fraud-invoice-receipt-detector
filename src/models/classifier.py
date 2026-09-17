import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np

class InvoiceFraudClassifier(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet18(weights=weights)
        num_features = self.backbone.fc.in_features
        # 2 classes: 0 = Genuine/Real, 1 = Manipulated/Fake
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, 2)
        )

    def forward(self, x):
        return self.backbone(x)

class FraudModelInference:
    def __init__(self, weights_path=None, device="cpu"):
        self.device = torch.device(device)
        self.model = InvoiceFraudClassifier(pretrained=(weights_path is None or not os.path.exists(weights_path)))
        if weights_path and os.path.exists(weights_path):
            state = torch.load(weights_path, map_location=self.device)
            self.model.load_state_dict(state)
            print(f"Loaded trained fraud model weights from {weights_path}")
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((448, 448)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def predict(self, image_path):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        pil_img = Image.open(image_path).convert("RGB")
        tensor = self.transform(pil_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]

        prob_real = float(probs[0])
        prob_fake = float(probs[1])

        return {
            "authenticity_score": round(prob_real * 100, 2), # 0-100%
            "tamper_probability": round(prob_fake * 100, 2), # 0-100%
            "prediction": "GENUINE" if prob_real >= 0.5 else "MANIPULATED",
            "is_manipulated": bool(prob_fake > 0.5)
        }

if __name__ == "__main__":
    inf = FraudModelInference()
    sample = r"E:\fraud Invoice&Receipt Detector\data\raw\invoice_0001.jpg"
    res = inf.predict(sample)
    print("Inference Test:", res)
