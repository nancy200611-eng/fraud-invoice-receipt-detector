import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np
import cv2

class SpatialChannelAttention(nn.Module):
    """
    Attention fusion block inspired by SW-MHSA in FUSION++:
    Applies spatial and channel-wise self-attention to focus on manipulation artifacts.
    """
    def __init__(self, channels):
        super().__init__()
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels, 1),
            nn.Sigmoid()
        )
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(channels, 1, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        ca = self.channel_gate(x)
        sa = self.spatial_gate(x)
        return x * ca * sa

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.net(x)

class FusionPlusModel(nn.Module):
    """
    FUSION++ Dual-Path Architecture (Thornton et al., 2025 & Aljuaid and Bhowmik, BMVC 2024)
    - Path A: RGB Visual Conv Stream
    - Path B: Multi-Filter Forensic Stream (Bayar, SRM, HOG, ELA)
    - Attention Block: Self-Attention Fusion
    - Multi-task Decoder:
        1. Anomaly / Authenticity Classifier (Real vs Manipulated)
        2. Pixel-wise Localization Map M
    """
    def __init__(self):
        super().__init__()
        # Path A: RGB Path (3 channels)
        self.rgb_conv1 = ConvBlock(3, 32)
        self.rgb_pool1 = nn.MaxPool2d(2)
        self.rgb_conv2 = ConvBlock(32, 64)
        self.rgb_pool2 = nn.MaxPool2d(2)
        self.rgb_conv3 = ConvBlock(64, 128)

        # Path B: Filters Path (4 channels: Bayar, SRM, HOG, ELA)
        self.flt_conv1 = ConvBlock(4, 32)
        self.flt_pool1 = nn.MaxPool2d(2)
        self.flt_conv2 = ConvBlock(32, 64)
        self.flt_pool2 = nn.MaxPool2d(2)
        self.flt_conv3 = ConvBlock(64, 128)

        # Attention Fusion: 128 + 128 = 256 channels
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(256, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.attention = SpatialChannelAttention(128)

        # Classification Head (Real vs Fake)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2)
        )

        # Decoder Head: Localization Map M (Upsampling)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec_conv1 = ConvBlock(64, 64)
        self.up2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec_conv2 = ConvBlock(32, 32)
        self.mask_out = nn.Conv2d(32, 1, 1)

    def forward(self, rgb_tensor, filter_tensor):
        # Path A
        r1 = self.rgb_conv1(rgb_tensor)
        r2 = self.rgb_conv2(self.rgb_pool1(r1))
        r3 = self.rgb_conv3(self.rgb_pool2(r2))

        # Path B
        f1 = self.flt_conv1(filter_tensor)
        f2 = self.flt_conv2(self.flt_pool1(f1))
        f3 = self.flt_conv3(self.flt_pool2(f2))

        # Fusion & Attention
        fused = torch.cat([r3, f3], dim=1) # [B, 256, H/4, W/4]
        f_feat = self.fusion_conv(fused)
        f_att = self.attention(f_feat) # F_fused

        # Classification Logits
        gap = self.avg_pool(f_att).flatten(1)
        logits = self.classifier(gap)

        # Localization Mask
        u1 = self.dec_conv1(self.up1(f_att))
        u2 = self.dec_conv2(self.up2(u1))
        mask_logits = self.mask_out(u2)
        mask = torch.sigmoid(mask_logits)

        return logits, mask

class FusionPlusInference:
    def __init__(self, weights_path=None, device="cpu"):
        self.device = torch.device(device)
        self.model = FusionPlusModel().to(self.device)
        if weights_path and os.path.exists(weights_path):
            try:
                state = torch.load(weights_path, map_location=self.device)
                self.model.load_state_dict(state)
                print(f"Loaded FUSION++ weights from {weights_path}")
            except Exception as e:
                print(f"Warning loading weights: {e}")
        self.model.eval()

    def prepare_tensors(self, image_path, target_size=(256, 256)):
        from src.forensics.image_forensics import DocumentForensics
        forensics = DocumentForensics()

        img_bgr = cv2.imread(image_path)
        img_resized = cv2.resize(img_bgr, target_size)
        rgb_norm = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        # Extract multi-filters
        bayar = cv2.resize(forensics.compute_bayar_residual(img_bgr)["bayar_map"], target_size).astype(np.float32) / 255.0
        srm = cv2.resize(forensics.compute_srm_residuals(img_bgr)["srm_map"], target_size).astype(np.float32) / 255.0
        hog = cv2.resize(forensics.compute_hog_residuals(img_bgr)["hog_map"], target_size).astype(np.float32) / 255.0
        ela = cv2.resize(cv2.cvtColor(forensics.compute_ela(image_path)["ela_map"], cv2.COLOR_RGB2GRAY), target_size).astype(np.float32) / 255.0

        # Stack filters: [4, H, W]
        filter_stack = np.stack([bayar, srm, hog, ela], axis=0)
        rgb_stack = np.transpose(rgb_norm, (2, 0, 1))

        rgb_t = torch.from_numpy(rgb_stack).float().unsqueeze(0).to(self.device)
        flt_t = torch.from_numpy(filter_stack).float().unsqueeze(0).to(self.device)
        return rgb_t, flt_t, img_bgr

    def predict(self, image_path):
        rgb_t, flt_t, orig_bgr = self.prepare_tensors(image_path)
        h_orig, w_orig = orig_bgr.shape[:2]

        with torch.no_grad():
            logits, mask = self.model(rgb_t, flt_t)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]
            pred_mask_np = mask.squeeze().cpu().numpy()

        prob_real = float(probs[0])
        prob_fake = float(probs[1])

        # Upscale predicted mask to original size
        mask_resized = cv2.resize((pred_mask_np * 255).astype(np.uint8), (w_orig, h_orig))
        _, binary_mask = cv2.threshold(mask_resized, 127, 255, cv2.THRESH_BINARY)

        # Enhanced Mask (morphological enhancement as per Figure 3 of paper)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
        enhanced_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

        return {
            "authenticity_score": round(prob_real * 100, 2),
            "tamper_probability": round(prob_fake * 100, 2),
            "is_manipulated": bool(prob_fake > 0.5),
            "predicted_mask": binary_mask,
            "enhanced_mask": enhanced_mask
        }

if __name__ == "__main__":
    model = FusionPlusModel()
    x_rgb = torch.randn(2, 3, 256, 256)
    x_flt = torch.randn(2, 4, 256, 256)
    logits, mask = model(x_rgb, x_flt)
    print("FUSION++ Architecture verified:")
    print("Logits shape:", logits.shape)
    print("Mask shape:", mask.shape)
