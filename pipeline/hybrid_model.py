"""
Novel Hybrid Deep Learning Model: ResNet-34 + CBAM (Convolutional Block Attention Module)
Fuses Spatial and Channel Attention for High-Accuracy Wafer Defect Pattern Classification.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34, ResNet34_Weights

from wafer_dataset import NUM_CLASSES

class ChannelAttention(nn.Module):
    """
    Channel Attention Module: Computes per-channel attention weights using AvgPool and MaxPool.
    """
    def __init__(self, in_channels: int, reduction_ratio: int = 16) -> None:
        super().__init__()
        reduced_channels = max(8, in_channels // reduction_ratio)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, reduced_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, in_channels, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return x * self.sigmoid(out)

class SpatialAttention(nn.Module):
    """
    Spatial Attention Module: Learns spatial defect location maps using 7x7 Convolution.
    """
    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(concat)
        return x * self.sigmoid(out)

class CBAMBlock(nn.Module):
    """
    Convolutional Block Attention Module combining Channel and Spatial Attention.
    """
    def __init__(self, channels: int, reduction_ratio: int = 16) -> None:
        super().__init__()
        self.channel_att = ChannelAttention(channels, reduction_ratio=reduction_ratio)
        self.spatial_att = SpatialAttention(kernel_size=7)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x

class ResNet34_CBAM_Hybrid(nn.Module):
    """
    Novel Hybrid Architecture: ResNet-34 backbone with integrated CBAM Attention Blocks.
    """
    def __init__(self, num_classes: int = NUM_CLASSES, pretrained: bool = True) -> None:
        super().__init__()
        weights = ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = resnet34(weights=weights)
        
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool

        # Stage Residual Layers
        self.layer1 = backbone.layer1
        self.cbam1 = CBAMBlock(64)

        self.layer2 = backbone.layer2
        self.cbam2 = CBAMBlock(128)

        self.layer3 = backbone.layer3
        self.cbam3 = CBAMBlock(256)

        self.layer4 = backbone.layer4
        self.cbam4 = CBAMBlock(512)

        self.avgpool = backbone.avgpool
        
        # Dual Feature Projection Classifier Head
        self.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.cbam1(x)

        x = self.layer2(x)
        x = self.cbam2(x)

        x = self.layer3(x)
        x = self.cbam3(x)

        x = self.layer4(x)
        x = self.cbam4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        logits = self.fc(x)
        return logits

def build_hybrid_model(num_classes: int = NUM_CLASSES, pretrained: bool = True) -> nn.Module:
    """Instantiate the novel ResNet34-CBAM Hybrid model."""
    return ResNet34_CBAM_Hybrid(num_classes=num_classes, pretrained=pretrained)

if __name__ == "__main__":
    print("[Hybrid Model Test] Testing forward pass shape...")
    dummy_input = torch.randn(4, 3, 224, 224)
    model = build_hybrid_model()
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output logits shape: {output.shape} (Expected: [4, {NUM_CLASSES}])")
    assert output.shape == (4, NUM_CLASSES)
    print("Success! ResNet34-CBAM Hybrid Model forward pass verified.")
