"""
Small heatmap-based tip refiner (Experiment 34/R2): given a small ROI
already roughly centered on the tip (by YOLO's coarse prediction), predicts
a precise heatmap for the true tip location within that ROI -- a much more
constrained task than the original full-gauge keypoint problem.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


class TipRefinerModel(nn.Module):
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

        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(576, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
        )
        self.final_conv = nn.Conv2d(32, 1, kernel_size=1)  # 56x56 output heatmap

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            cutoff = len(self.features) - self.n_unfrozen_blocks
            for i, layer in enumerate(self.features):
                if i < cutoff:
                    layer.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)  # (batch, 576, 7, 7)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        return self.final_conv(x)  # (batch, 1, 56, 56)