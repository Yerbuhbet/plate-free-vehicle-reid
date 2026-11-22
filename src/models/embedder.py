from pathlib import Path
from typing import Dict, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from transformers import AutoModel


class DINOv2Embedder(nn.Module):
    def __init__(
        self,
        weights_path: Optional[str | Path] = None,
        backbone_name: str = "facebook/dinov2-base",
        use_fp16: bool = True,
        device: Optional[str] = None,
    ):
        super().__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_fp16 = use_fp16 and (self.device == "cuda")

        self.backbone = AutoModel.from_pretrained(backbone_name)
        self.bnneck = nn.BatchNorm1d(self.backbone.config.hidden_size)
        self.bnneck.bias.requires_grad_(False)

        if weights_path and Path(weights_path).exists():
            state = torch.load(weights_path, map_location="cpu")
            self.load_state_dict(state, strict=True)

        self.to(self.device)
        if self.use_fp16:
            self.half()
        else:
            self.float()
        self.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    @torch.inference_mode()
    def encode_image(self, img: Image.Image) -> np.ndarray:
        x = self.transform(img.convert("RGB")).unsqueeze(0).to(self.device)
        if self.use_fp16:
            x = x.half()
        feat = self.backbone(pixel_values=x).last_hidden_state[:, 0]
        bn_feat = self.bnneck(feat)
        emb = F.normalize(bn_feat.float(), p=2, dim=1)
        return emb.squeeze(0).cpu().numpy().astype(np.float32)

    def encode_multiview(
        self,
        views: Dict[str, Image.Image],
        weights: Optional[Dict[str, float]] = None,
    ) -> np.ndarray:
        """Агрегирует до 4 ракурсов (front, rear, left, right) в единый вектор."""
        default_w = {"front": 0.35, "left": 0.25, "right": 0.25, "rear": 0.15}
        w_map = weights or default_w

        acc = None
        total_w = 0.0
        for view_name, img in views.items():
            w = w_map.get(view_name, 0.25)
            vec = self.encode_image(img) * w
            acc = vec if acc is None else acc + vec
            total_w += w

        fused = acc / max(total_w, 1e-6)
        return (fused / (np.linalg.norm(fused) + 1e-12)).astype(np.float32)
