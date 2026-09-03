"""
Two-stage crop-based heatmap keypoint model (DL architecture, thrid attempt).

Stage 1 (outside this model): crop the image to the gauge region before this model see it -- ground-truth
bbox during training/evaluation here; the classical baseline's find_gauge_circle is the natural Stage-1
"detector" for real depolyment. This tests whether earlier model's poor performance was a full-scene localization
problem, not a landmark-learning problem.

Improvements over heatmap_model.py:
- Higher-resolution decoder (7 -> 14 -> 28 -> 56, one more upsampling stage)
- One skip connection from an earlier, higher-resolution backbone stage (lightweight FPN-style lateral connection)
for local precision
- Soft-argmax decoding (differentiable spatial expectation), used both for a coordinate loss term during training
and for inference
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

N_KEYPOINTS = 4
HEATMAP_SIZE = 56

def _find_skip_layer_index(features: nn.Sequential, input_size: int = 224, target_spatial: int = 14) -> int:
    """Dry forward pass to find which block first produces a target_spatial x target_spatial map."""
    dummy = torch.zeros(1, 3, input_size, input_size)
    x = dummy
    with torch.no_grad():
        for i, layer in enumerate(features):
            x = layer(x)
            if x.shape[-1] == target_spatial:
                return i
    raise RuntimeError(f"No layer produced a {target_spatial}x{target_spatial} feature map.")

class CropHeatmapModel(nn.Module):
    def __init__(self, freeze_backbone: bool = True, n_unfrozen_blocks: int = 1):
        super().__init__()
        self.freeze_backbone = freeze_backbone
        self.n_unfrozen_blocks = n_unfrozen_blocks

        weights =MobileNet_V3_Small_Weights.DEFAULT
        backbone = mobilenet_v3_small(weights=weights)
        self.features = backbone.features

        if freeze_backbone:
            for param in self.features.parameters():
                param.requires_grad = False
            for layer in self.features[-n_unfrozen_blocks:]:
                for param in layer.parameters():
                    param.requires_grad = True

        self.skip_index = _find_skip_layer_index(self.features)

        dummy = torch.zeros(1, 3, 224, 224)
        skip_channels = None
        with torch.no_grad():
            x = dummy
            for i, layer in enumerate(self.features):
                x = layer(x)
                if i == self.skip_index:
                    skip_channels = x.shape[1]
                    break

        backbone_channels = 576
        self.skip_proj = nn.Conv2d(skip_channels, 128, kernel_size=1)

        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(backbone_channels, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )   # 7 -> 14, merged with skip here
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )   # 14 -> 28
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )   # 28 -> 56
        self.final_conv = nn.Conv2d(32, N_KEYPOINTS, kernel_size=1)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            cutoff = len(self.features) - self.n_unfrozen_blocks
            for i, layer in enumerate(self.features):
                if i < cutoff:
                    layer.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip_feat = None
        out = x
        for i, layer in enumerate(self.features):
            out = layer(out)
            if i == self.skip_index:
                skip_feat = out

        x = self.up1(out)
        skip_proj = self.skip_proj(skip_feat)
        if skip_proj.shape[-2:] != x.shape[-2:]:
            skip_proj = F.interpolate(skip_proj, size=x.shape[-2:], mode="bilinear", align_corners=False)
        x = x + skip_proj

        x = self.up2(x)
        x = self.up3(x)
        x = self.final_conv(x)
        return x

def soft_argmax_decode(heatmap: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """
    Differentiable coordinate decoding: treats each keypoint's heatmap as a spatial probability distribution
    (via softmax) and computes the expected (x, y) position -- a weighted average, not a hard peak. Usable both
    inside taining's loss and for final inference.
    """
    batch_size, n_keypoints, h, w = heatmap.shape
    flat = heatmap.view(batch_size, n_keypoints, -1) * temperature
    weights = F.softmax(flat, dim=2).view(batch_size, n_keypoints, h, w)

    device = heatmap.device
    y_coords = torch.linspace(0, 1, h, device=device).view(1, 1, h, 1).expand(batch_size, n_keypoints, h, w)
    x_coords = torch.linspace(0, 1, w, device=device).view(1, 1, 1, w).expand(batch_size, n_keypoints, h, w)

    expected_x = (weights * x_coords).sum(dim=(2, 3))
    expected_y = (weights * y_coords).sum(dim=(2, 3))

    coords = torch.stack([expected_x, expected_y], dim=2)
    return coords.view(batch_size, n_keypoints * 2)

def hard_argmax_decode(heatmap: torch.Tensor) -> torch.Tensor:
    """Non-differentiable decode: each keypoint's peak (brightest) pixel directly, no averaging."""
    batch_size, n_keypoints, h, w = heatmap.shape
    flat = heatmap.view(batch_size, n_keypoints, -1)
    peak_indices = flat.argmax(dim=2)
    peak_y = (peak_indices // w).float() / (h - 1)
    peak_x = (peak_indices % w).float() / (w - 1)
    coords = torch.stack([peak_x, peak_y], dim=2)
    return coords.view(batch_size, n_keypoints * 2)