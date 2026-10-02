"""
Task 2: Corruption classifier (4-class: clean, salt_pepper, blur, occlusion).
A straightforward convolutional classifier used to route inputs to the
correct specialist autoencoder during hard-routed inference.
"""
import torch.nn as nn


class CorruptionClassifier(nn.Module):
    def __init__(self, base_ch=32, dropout=0.2, n_classes=4):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, base_ch, 4, 2, 1), nn.BatchNorm2d(base_ch), nn.ReLU(inplace=True),        # 128->64
            nn.Conv2d(base_ch, base_ch*2, 4, 2, 1), nn.BatchNorm2d(base_ch*2), nn.ReLU(inplace=True),  # 64->32
            nn.Conv2d(base_ch*2, base_ch*4, 4, 2, 1), nn.BatchNorm2d(base_ch*4), nn.ReLU(inplace=True),# 32->16
            nn.Conv2d(base_ch*4, base_ch*8, 4, 2, 1), nn.BatchNorm2d(base_ch*8), nn.ReLU(inplace=True),# 16->8
            nn.AdaptiveAvgPool2d(1),  # -> base_ch*8 x 1 x 1, robust to input size
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(base_ch*8, base_ch*4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(base_ch*4, n_classes),
        )

    def forward(self, x):
        features = self.conv(x)
        return self.classifier(features)  # raw logits, apply softmax/argmax outside
