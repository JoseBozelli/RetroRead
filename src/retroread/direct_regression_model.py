"""
Direct end-to-end numeric regression model (Experiment 28): image -> 
single number, no geometry at all. Predicts normalized position within 
the gauge's own scale range (0-1), since raw value regression is 
meaningless across gauges with different randomized ranges -- 
this is the closest honest approximation to "pure regression"
given the dataset, and is expected to perform poorly (the whole point of
including it: showing WHY geometry-aware approaches are preferred).
"""

import torch
import torch.nn as nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


class DirectRegressionModel(nn.Module):
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
            nn.Linear(128, 1),
            nn.Sigmoid(),  # normalized position in [0,1] within the gauge's own scale range
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
        return self.head(x)  # (batch, 1)