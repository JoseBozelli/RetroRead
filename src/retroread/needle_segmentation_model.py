"""
Needle segmentation + center heatmap model (Experiment 22).

Two output heads sharing a common decoder: needle segmentation mask (fine
112x112 resolution, for robust geometric line-fitting across many pixels
rather than one predicted point) and a gauge-center heatmap (56x56,
matching crop_heatmap_model's proven resolution). The final reading is
computed geometrically in needle_segmentation_geometry.py -- not predicted
directly by the network.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


def _find_skip_layer_index(features: nn.Sequential, input_size: int = 224, target_spatial: int = 14) -> int:
    dummy = torch.zeros(1, 3, input_size, input_size)
    x = dummy
    with torch.no_grad():
        for i, layer in enumerate(features):
            x = layer(x)
            if x.shape[-1] == target_spatial:
                return i
    raise RuntimeError(f"No layer produced a {target_spatial}x{target_spatial} feature map.")


class NeedleSegmentationModel(nn.Module):
    def __init__(self, freeze_backbone: bool = True, n_unfrozen_blocks: int = 1):
        super().__init__()
        self.freeze_backbone = freeze_backbone
        self.n_unfrozen_blocks = n_unfrozen_blocks

        weights = MobileNet_V3_Small_Weights.DEFAULT
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
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
        )  # 7 -> 14, merge skip here
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        )  # 14 -> 28
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
        )  # 28 -> 56

        self.center_head = nn.Conv2d(32, 1, kernel_size=1)  # branches off at 56x56

        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16), nn.ReLU(inplace=True),
        )  # 56 -> 112, mask head continues for finer resolution
        self.mask_head = nn.Conv2d(16, 1, kernel_size=1)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            cutoff = len(self.features) - self.n_unfrozen_blocks
            for i, layer in enumerate(self.features):
                if i < cutoff:
                    layer.eval()
        return self

    def forward(self, x: torch.Tensor):
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

        center_logits = self.center_head(x)

        x_mask = self.up4(x)
        mask_logits = self.mask_head(x_mask)

        return mask_logits, center_logits