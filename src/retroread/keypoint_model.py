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
        self.pool = nn.AdaptiveAvgPool2d((4,4))     # preserve a coarse spatial grid, not a single point

        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            for param in self.features.parameters():
                param.requires_grad = False
            # Partially unfreeze the last block, so pretrained features can adapt slightly toward this
            # specific localization task.
            for param in self.features[-1].parameters():
                param.requires_grad = True

        backbone_out_channels = 576 * 4 * 4     # MobileNetV3-Small's feature output width. 
                                                # 576 channels x 4x4 spatial grid, flattened

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(backbone_out_channels, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, N_OUTPUTS),
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
        Keeps all backbone layers except the last block in eval mode (stable pretrained BatchNorm statistics),
        since they are frozen. The last block is now trainable, so it follows normal train/eval toggling.
        """
        super().train(mode)
        if self.freeze_backbone:
            for i, layer in enumerate(self.features):
                if i != len(self.features) -1:
                    layer.eval()
        return self