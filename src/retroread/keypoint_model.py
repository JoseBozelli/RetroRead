"""
Deep learning keypoint model for gauge geometry. Predicts 4 keypoints (dial_center, dial_tip, dial_min, and dial_max)
as normalized (x, y) coordinates in [0, 1], via a frozen pretrained MobileNetV3-Small backbone plus a small trainable
regression head.

Backbone is frozen (not fine-tuned) to keep training feasible on CPU -- only the head's parameters receive gradient
updates.
"""

import torch
import torch.nn as nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

N_KEYPOINTS = 4
N_OUTPUTS = N_KEYPOINTS * 2 # (x, y) per keypoint

class GaugeKeypointModel (nn.Module):
    def __init__(self, freeze_backbone: bool = True):
        super().__init__()

        weights = MobileNet_V3_Small_Weights.DEFAULT
        backbone = mobilenet_v3_small(weights=weights)

        # Keep only the convolutional feature extractor, drop the original ImageNet classification head.
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            for param in self.features.parameters():
                param.requires_grad = False

        backbone_out_channels = 576     # MobileNetV3-Small's feature output width

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(backbone_out_channels, 128),
            nn.ReLU(),
            nn.Linear(128, N_OUTPUTS),
            nn.Sigmoid(),   # constrains output to [0, 1], matching normalized coordinates
        )

        # Preprocessing transform matching what the pretrained weights expect.
        self.preprocess = weights.transforms()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = self.head(x)
        return x    # shape: (batch_size, 8) -- [cx, cy, tx, ty, minx, miny, maxx, maxy]

    def train(self, mode: bool = True):
        """
        Overrides the default train/eval toggle: when the backbone is frozen, 
        it must always stay in eval mode -- otherwise its BatchNorm layers use noisy per-batch 
        statistics instead of the stable, pretrained running averages, even though its weights
        aren't being updated. Only the head toggles normally.
        """
        super().train(mode)
        if self.freeze_backbone:
            self.features.eval()
        return self