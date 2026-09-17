import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

class GradCAM:
    """
    Grad-CAM (Gradient-weighted Class Activation Mapping)
    Highlights which regions of the invoice/receipt triggered the 'Manipulated' prediction.
    """
    def __init__(self, model, target_layer=None):
        self.model = model
        self.model.eval()
        
        # Default to the final conv layer of resnet18
        if target_layer is None:
            self.target_layer = self.model.backbone.layer4[-1].conv2
        else:
            self.target_layer = target_layer

        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_heatmap(self, input_tensor, target_class=1):
        """
        input_tensor: [1, 3, H, W]
        target_class: 1 for Manipulated/Fake, 0 for Genuine
        """
        self.model.zero_grad()
        logits = self.model(input_tensor)
        
        # Score for target class
        score = logits[0, target_class]
        score.backward()

        # Global average pooling of gradients
        # gradients shape: [1, 512, h, w]
        weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)

        # Weighted combination of activation maps
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)

        # Normalize between 0 and 1
        cam_min, cam_max = torch.min(cam), torch.max(cam)
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        else:
            cam = torch.zeros_like(cam)

        cam_np = cam.squeeze().cpu().numpy()
        return cam_np

    def overlay_on_image(self, original_image_path, cam_np, alpha=0.5):
        orig_bgr = cv2.imread(original_image_path)
        h, w = orig_bgr.shape[:2]

        # Resize CAM to original image dimensions
        cam_resized = cv2.resize(cam_np, (w, h))
        heatmap_color = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)

        overlay = cv2.addWeighted(heatmap_color, alpha, orig_bgr, 1 - alpha, 0)
        return overlay, cam_resized

if __name__ == "__main__":
    from src.models.classifier import FraudModelInference
    inf = FraudModelInference()
    gradcam = GradCAM(inf.model)

    sample_img = r"E:\fraud Invoice&Receipt Detector\data\manipulated\images\fake_invoice_0001.jpg"
    pil_img = Image.open(sample_img).convert("RGB")
    tensor = inf.transform(pil_img).unsqueeze(0)

    cam_map = gradcam.generate_heatmap(tensor, target_class=1)
    overlay, resized = gradcam.overlay_on_image(sample_img, cam_map)

    out_vis = r"E:\fraud Invoice&Receipt Detector\data\samples\sample_gradcam.jpg"
    cv2.imwrite(out_vis, overlay)
    print("Grad-CAM generated and saved to:", out_vis)
