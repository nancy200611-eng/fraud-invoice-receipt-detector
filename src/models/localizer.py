import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class ForensicTamperLocalizer(nn.Module):
    """
    Advanced Phase 3: Encoder-Decoder Architecture (Dual-stream / FUSION-style)
    Fuses RGB image with 3-channel forensic residuals (ELA + SRM + Noise)
    to output pixel-wise tampering localization mask [H, W].
    """
    def __init__(self, in_channels=6): # 3 RGB + 3 Forensic
        super().__init__()
        # Encoder
        self.inc = DoubleConv(in_channels, 32)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(32, 64))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))

        # Decoder with skip connections
        self.up1 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv_up1 = DoubleConv(256, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv_up2 = DoubleConv(128, 64)

        self.up3 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.conv_up3 = DoubleConv(64, 32)

        # Output head: 1-channel binary mask logit
        self.outc = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x = self.up1(x4)
        x = torch.cat([x, x3], dim=1)
        x = self.conv_up1(x)

        x = self.up2(x)
        x = torch.cat([x, x2], dim=1)
        x = self.conv_up2(x)

        x = self.up3(x)
        x = torch.cat([x, x1], dim=1)
        x = self.conv_up3(x)

        logits = self.outc(x)
        return torch.sigmoid(logits)

def predict_tamper_mask(image_path, model=None, device="cpu"):
    """
    Inference helper: constructs RGB + Forensic feature tensor,
    runs localizer, and returns binary mask & overlay.
    """
    from src.forensics.image_forensics import DocumentForensics
    forensics = DocumentForensics()

    img_bgr = cv2.imread(image_path)
    h_orig, w_orig = img_bgr.shape[:2]

    # Compute forensic channels
    ela_res = forensics.compute_ela(image_path)
    srm_res = forensics.compute_srm_residuals(img_bgr)
    noise_res = forensics.compute_noise_inconsistency(img_bgr)

    # Resize all to 256x256 for network
    target_size = (256, 256)
    img_rgb = cv2.cvtColor(cv2.resize(img_bgr, target_size), cv2.COLOR_BGR2RGB) / 255.0
    ela_norm = cv2.resize(ela_res["ela_map"], target_size) / 255.0
    srm_norm = np.expand_dims(cv2.resize(srm_res["srm_map"], target_size), -1) / 255.0
    noise_norm = np.expand_dims(cv2.resize(noise_res["noise_map"], target_size), -1) / 255.0

    # Composite 6-channel input
    forensic_stack = np.concatenate([img_rgb, ela_norm[:, :, :1], srm_norm, noise_norm], axis=-1)
    tensor = torch.from_numpy(forensic_stack).permute(2, 0, 1).float().unsqueeze(0).to(device)

    if model is None:
        model = ForensicTamperLocalizer().to(device)
        model.eval()

    with torch.no_grad():
        pred_mask = model(tensor).squeeze().cpu().numpy()

    # Upsample back to original size
    mask_full = cv2.resize((pred_mask > 0.4).astype(np.uint8) * 255, (w_orig, h_orig))
    
    # Overlay in Red
    overlay = img_bgr.copy()
    overlay[mask_full > 0] = [0, 0, 255] # Red highlight on tampered pixels
    blended = cv2.addWeighted(overlay, 0.4, img_bgr, 0.6, 0)

    return mask_full, blended

if __name__ == "__main__":
    sample = r"E:\fraud Invoice&Receipt Detector\data\manipulated\images\fake_invoice_0001.jpg"
    mask, overlay = predict_tamper_mask(sample)
    out_mask = r"E:\fraud Invoice&Receipt Detector\data\samples\sample_localizer_mask.png"
    out_overlay = r"E:\fraud Invoice&Receipt Detector\data\samples\sample_localizer_overlay.jpg"
    cv2.imwrite(out_mask, mask)
    cv2.imwrite(out_overlay, overlay)
    print("Localizer mask and overlay saved.")
