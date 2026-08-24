"""16-band Hyperspectral Faster R-CNN with Spectral Attention Stem."""

import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from . import config


class SpectralChannelAttention(nn.Module):
    """Squeeze-and-Excitation along spectral bands to capture material reflectance signatures."""
    def __init__(self, in_channels: int = 16, reduction: int = 4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SpectralStem(nn.Module):
    """Integrates Spectral Attention directly into the 16-channel input stem."""
    def __init__(self, original_conv1: nn.Conv2d, in_bands: int = 16):
        super().__init__()
        self.spectral_attn = SpectralChannelAttention(in_channels=in_bands)
        self.conv1 = nn.Conv2d(
            in_bands,
            original_conv1.out_channels,
            kernel_size=original_conv1.kernel_size,
            stride=original_conv1.stride,
            padding=original_conv1.padding,
            bias=(original_conv1.bias is not None),
        )
        with torch.no_grad():
            avg_weight = original_conv1.weight.data.mean(dim=1, keepdim=True)
            self.conv1.weight.data = avg_weight.repeat(1, in_bands, 1, 1) / in_bands * 3
            if original_conv1.bias is not None:
                self.conv1.bias.data = original_conv1.bias.data.clone()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.spectral_attn(x)
        return self.conv1(x)


def build_model(num_classes: int = None, pretrained: bool = True) -> FasterRCNN:
    num_classes = num_classes or (config.NUM_CLASSES + 1)

    image_mean = [0.5] * config.NUM_BANDS
    image_std = [0.5] * config.NUM_BANDS

    # Load complete COCO pretrained Faster R-CNN (Backbone + FPN + RPN + RoI Head)
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights="DEFAULT" if pretrained else None,
        image_mean=image_mean,
        image_std=image_std,
        min_size=300,
        max_size=600,
    )

    old_conv1 = model.backbone.body.conv1
    model.backbone.body.conv1 = SpectralStem(old_conv1, in_bands=config.NUM_BANDS)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


if __name__ == "__main__":
    model = build_model()
    model.eval()
    dummy = [torch.randn(16, 256, 256)]
    with torch.no_grad():
        out = model(dummy)
    print("Pretrained base model verified successfully.")