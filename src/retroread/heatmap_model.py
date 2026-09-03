"""
Heatmap-based keyopint model (Deep Learning, second architecture).

Unlike the first attempt (keypoint_model.py), this preserves spatial structure throughout: instead of flattening
features into a coordinate regression, the head unsamples the backbone's spatial feature map back up and outputs
one small heatmap per keypoint -- a 2D "likelihood map" of where that point is. The predicted coordinate is read 
off as the peak (brightest pixel) of its heatmap. This is the standard approach used by most real keypoint/pose-
estimation systems, pecifically because flatten-based regression discards positional information (see docs/
decision_log.md for the diagnosis that motivated this rebuild).
"""

import torch
import torch.nn as nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

N_KEYPOINTS = 4
HEATMAP_SIZE = 28   # output heatmap resolution (28x28 per keypoint)

class GaugeHeatmapModel(nn.Module):
    def __init__(self, freeze_backbone: bool = True):
        super().__init__()
        self.freeze_backbone = freeze_backbone

        weights = MobileNet_V3_Small_Weights.DEFAULT
        backbone = mobilenet_v3_small(weights=weights)
        self.features = backbone.features       # spatial output, NOT pooled/flattened

        if freeze_backbone:
            for param in self.features.parameters():
                param.requires_grad = False
            # Last block stays trainable, as in the previous attempt --
            # lets pretrained features adapt slightly to this task.
            for param in self.features[-1].parameters():
                param.requires_grad = True

        backbone_channels = 576

        # Upsampling head: backbone output is roughly 7x7 for a 224x224
        # input (MobileNetV3's total stride is 32). Two upsampling stages
        # (7 -> 14 -> 28) bring it back to a usable spatial resolution.
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(backbone_channels, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, N_KEYPOINTS, kernel_size=1),      # one output channel per keypoint
        )

        self.preprocess = weights.transforms()      # overridden by heatmap_dataset's resize-only version

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            for i, layer in enumerate(self.features):
                if i != len(self.features) -1:
                    layer.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)        # (batch, 576, 7, 7)
        x = self.upsample(x)        # (batch, N_KEYPOINTS, 28, 28)
        return (x)

def decode_heatmap(heatmap: torch.Tensor) -> torch.Tensor:
    """
    Convert a predicted heatmap batch (batch, N_KEYPOINTS, H, W) into normalized (x, y) coordinates per
    keypoint, by finding each channel's peak (brighest pixel) and dividing by the grid size.

    Returns: (batch, N_KEYPOINTS * 2) tensor, same order as heatmap channels, ready to compare against
    the same [cx, cy, tx, ty, minx, miny, maxx, maxy]-style target ordering used elsewhere.
    """
    batch_size, n_keypoints, h, w = heatmap.shape
    flat = heatmap.view(batch_size, n_keypoints, -1)
    peak_indices = flat.argmax(dim=2)   # (batch, n_keypoints)

    peak_y = (peak_indices // w).float() / h
    peak_x = (peak_indices % w).float() / w

    coords = torch.stack([peak_x, peak_y], dim=2)   # (batch, n_keypoints, 2)
    return coords.view(batch_size, n_keypoints * 2)