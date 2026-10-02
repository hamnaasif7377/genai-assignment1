"""
Task 1: Universal Multi-Corruption Denoising Autoencoder.
Convolutional encoder -> genuine compressed bottleneck -> convolutional decoder.
No skip connections, so the bottleneck is a real information constraint.
"""
import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, base_ch=32, bottleneck_dim=256, dropout=0.1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, base_ch, 4, 2, 1), nn.BatchNorm2d(base_ch), nn.ReLU(inplace=True),
            nn.Conv2d(base_ch, base_ch * 2, 4, 2, 1), nn.BatchNorm2d(base_ch * 2), nn.ReLU(inplace=True),
            nn.Conv2d(base_ch * 2, base_ch * 4, 4, 2, 1), nn.BatchNorm2d(base_ch * 4), nn.ReLU(inplace=True),
            nn.Conv2d(base_ch * 4, base_ch * 8, 4, 2, 1), nn.BatchNorm2d(base_ch * 8), nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
        )
        self.flatten_dim = base_ch * 8 * 8 * 8
        self.fc = nn.Linear(self.flatten_dim, bottleneck_dim)

    def forward(self, x):
        x = self.conv(x)
        x = x.flatten(1)
        return self.fc(x)


class Decoder(nn.Module):
    def __init__(self, base_ch=32, bottleneck_dim=256, dropout=0.1):
        super().__init__()
        self.base_ch = base_ch
        self.fc = nn.Linear(bottleneck_dim, base_ch * 8 * 8 * 8)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(base_ch * 8, base_ch * 4, 4, 2, 1), nn.BatchNorm2d(base_ch * 4), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 4, 2, 1), nn.BatchNorm2d(base_ch * 2), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base_ch * 2, base_ch, 4, 2, 1), nn.BatchNorm2d(base_ch), nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.ConvTranspose2d(base_ch, 3, 4, 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, self.base_ch * 8, 8, 8)
        return self.deconv(x)


class UniversalAutoencoder(nn.Module):
    def __init__(self, base_ch=32, bottleneck_dim=256, dropout=0.1):
        super().__init__()
        self.encoder = Encoder(base_ch, bottleneck_dim, dropout)
        self.decoder = Decoder(base_ch, bottleneck_dim, dropout)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)
