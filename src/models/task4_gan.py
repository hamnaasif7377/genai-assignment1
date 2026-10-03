"""
Task 4: Conditional GAN for style-conditioned face-to-sketch generation.
- Generator: U-Net encoder-decoder with skip connections, conditioned on
  a learned style embedding injected at the bottleneck. Uses resize-
  convolution (upsample + conv) instead of ConvTranspose2d in the decoder
  to avoid checkerboard/speckle artifacts common with transposed convolutions.
- Discriminator: PatchGAN, also conditioned on the style embedding,
  classifying local patches of the (photo, sketch) pair as real/fake.
"""
import torch
import torch.nn as nn


def down_block(in_ch, out_ch, use_norm=True):
    layers = [nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=not use_norm)]
    if use_norm:
        layers.append(nn.BatchNorm2d(out_ch))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    return nn.Sequential(*layers)


def up_block(in_ch, out_ch, use_dropout=False):
    """Resize-convolution: nearest-neighbor upsample + regular conv,
    instead of ConvTranspose2d, to avoid checkerboard artifacts."""
    layers = [
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=False),
        nn.BatchNorm2d(out_ch),
    ]
    if use_dropout:
        layers.append(nn.Dropout(0.5))
    layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class UNetGenerator(nn.Module):
    def __init__(self, base_ch=64, n_styles=3, style_dim=16, dropout=True):
        super().__init__()
        self.style_embedding = nn.Embedding(n_styles, style_dim)

        self.enc1 = down_block(3, base_ch, use_norm=False)          # 128->64
        self.enc2 = down_block(base_ch, base_ch * 2)                # 64->32
        self.enc3 = down_block(base_ch * 2, base_ch * 4)            # 32->16
        self.enc4 = down_block(base_ch * 4, base_ch * 8)            # 16->8
        self.enc5 = down_block(base_ch * 8, base_ch * 8)            # 8->4

        self.style_proj = nn.Conv2d(base_ch * 8 + style_dim, base_ch * 8, kernel_size=1)

        self.dec5 = up_block(base_ch * 8, base_ch * 8, use_dropout=dropout)
        self.dec4 = up_block(base_ch * 8 * 2, base_ch * 4, use_dropout=dropout)
        self.dec3 = up_block(base_ch * 4 * 2, base_ch * 2)
        self.dec2 = up_block(base_ch * 2 * 2, base_ch)
        self.dec1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(base_ch * 2, 3, 3, 1, 1),
            nn.Tanh(),
        )

    def forward(self, photo, style):
        e1 = self.enc1(photo)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)

        style_vec = self.style_embedding(style)
        style_map = style_vec[:, :, None, None].expand(-1, -1, e5.size(2), e5.size(3))
        bottleneck = torch.cat([e5, style_map], dim=1)
        bottleneck = self.style_proj(bottleneck)

        d5 = self.dec5(bottleneck)
        d4 = self.dec4(torch.cat([d5, e4], dim=1))
        d3 = self.dec3(torch.cat([d4, e3], dim=1))
        d2 = self.dec2(torch.cat([d3, e2], dim=1))
        out = self.dec1(torch.cat([d2, e1], dim=1))

        return out


class PatchGANDiscriminator(nn.Module):
    def __init__(self, base_ch=64, n_styles=3, style_dim=16):
        super().__init__()
        self.style_embedding = nn.Embedding(n_styles, style_dim)
        in_ch = 3 + 3 + style_dim

        self.model = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_ch, base_ch * 2, 4, 2, 1), nn.BatchNorm2d(base_ch * 2), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_ch * 2, base_ch * 4, 4, 2, 1), nn.BatchNorm2d(base_ch * 4), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_ch * 4, base_ch * 8, 4, 1, 1), nn.BatchNorm2d(base_ch * 8), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_ch * 8, 1, 4, 1, 1),
        )

    def forward(self, photo, sketch, style):
        style_vec = self.style_embedding(style)
        style_map = style_vec[:, :, None, None].expand(-1, -1, photo.size(2), photo.size(3))
        x = torch.cat([photo, sketch, style_map], dim=1)
        return self.model(x)
