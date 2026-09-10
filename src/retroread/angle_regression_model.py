"""
sin/cos angle regression model: predicts (sin theta, cos theta) directly rather than a coordinate -- avoids the
wraparound discontinuity of raw angle regression (359 deg and 1 deg look numerically far apart despite being nearly
identical). Global pooling, not a spatial decoder -- angle is a single scalar quantity, not a spatial map, so this
deliberately does NOT reuse the heatmap architecture.
"""

import torch
import torch.nn as nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

class AngleRegressionModel (nn.Module):
    def __init__(self, freeze_backbone: bool = True, n_unfrozen_blocks: int = 1):
        super().__init__()
        self.freeze_backbone = freeze_backbone
        self.n_unfrozen_blocks = n_unfrozen_blocks

        weights = MobileNet_V3_Small_Weights.DEFAULT
        backbone = mobilenet_v3_small(weights=weights)
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)

        if freeze_backbone:
            for param in self.features.parameters():
                param.requires_grad = False
            for layer in self.features[-n_unfrozen_blocks:]:
                for param in layer.parameters():
                    param.requires_grad = True

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(576, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2),       # sin, cos
            nn.Tanh(),  # bound to [-1, 1], matching real sin/cos range -- stabilizes
        )
        self.preprocess = weights.transforms()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            cutoff = len(self.features) - self.n_unfrozen_blocks
            for i, layer in enumerate(self.features):
                if i < cutoff:
                    layer.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return self.head(x)     # (batch, 2)    -- raw sin, cos; atan2 handles unnormalized values fine