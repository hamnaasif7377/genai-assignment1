"""
Task 4: Conditional GAN for style-conditioned face-to-sketch generation.
- Generator: U-Net encoder-decoder with skip connections, conditioned on
  a learned style embedding injected at the bottleneck.
- Discriminator: PatchGAN, also conditioned on the style embedding,
  classifying local patches of the (photo, sketch) pair as real/fake.
"""
import torch
import torch.nn as nn


def conv_block(in_ch, out_ch, down=True, use_norm=True, use_dropout=False):
    layers = []
    if down:
        layers.append(nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=not use_norm))
    else:
        layers.append(nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1, bias=not use_norm))
    if use_norm:
        layers.append(nn.BatchNorm2d(out_ch))
    if use_dropout:
        layers.append(nn.Dropout(0.5))
    layers.append(nn.LeakyReLU(0.2, inplace=True) if down else nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class UNetGenerator(nn.Module):
    """
    U-Net with skip connections. Style condition is a learned embedding,
    broadcast and concatenated with the bottleneck features.
    Input: 128x128 RGB photo -> Output: 128x128 RGB sketch.
    """
    def __init__(self, base_ch=64, n_styles=3, style_dim=16, dropout=True):
        super().__init__()
        self.style_embedding = nn.Embedding(n_styles, style_dim)

        # Encoder: 128 -> 64 -> 32 -> 16 -> 8 -> 4
        self.enc1 = conv_block(3, base_ch, down=True, use_norm=False)          # 128->64
        self.enc2 = conv_block(base_ch, base_ch * 2, down=True)                # 64->32
        self.enc3 = conv_block(base_ch * 2, base_ch * 4, down=True)            # 32->16
        self.enc4 = conv_block(base_ch * 4, base_ch * 8, down=True)            # 16->8
        self.enc5 = conv_block(base_ch * 8, base_ch * 8, down=True)            # 8->4

        # Style injected here, at the bottleneck, via a 1x1 conv after concatenation
        self.style_proj = nn.Conv2d(base_ch * 8 + style_dim, base_ch * 8, kernel_size=1)

        # Decoder: 4 -> 8 -> 16 -> 32 -> 64 -> 128, with skip connections
        self.dec5 = conv_block(base_ch * 8, base_ch * 8, down=False, use_dropout=dropout)
        self.dec4 = conv_block(base_ch * 8 * 2, base_ch * 4, down=False, use_dropout=dropout)
        self.dec3 = conv_block(base_ch * 4 * 2, base_ch * 2, down=False)
        self.dec2 = conv_block(base_ch * 2 * 2, base_ch, down=False)
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(base_ch * 2, 3, 4, 2, 1),
            nn.Tanh(),  # output in [-1, 1], matches dataset normalization
        )

    def forward(self, photo, style):
        e1 = self.enc1(photo)   # base_ch,     64x64
        e2 = self.enc2(e1)      # base_ch*2,   32x32
        e3 = self.enc3(e2)      # base_ch*4,   16x16
        e4 = self.enc4(e3)      # base_ch*8,   8x8
        e5 = self.enc5(e4)      # base_ch*8,   4x4

        style_vec = self.style_embedding(style)                       # [B, style_dim]
        style_map = style_vec[:, :, None, None].expand(-1, -1, e5.size(2), e5.size(3))
        bottleneck = torch.cat([e5, style_map], dim=1)
        bottleneck = self.style_proj(bottleneck)

        d5 = self.dec5(bottleneck)                   # base_ch*8, 8x8
        d4 = self.dec4(torch.cat([d5, e4], dim=1))    # base_ch*4, 16x16
        d3 = self.dec3(torch.cat([d4, e3], dim=1))    # base_ch*2, 32x32
        d2 = self.dec2(torch.cat([d3, e2], dim=1))    # base_ch,   64x64
        out = self.dec1(torch.cat([d2, e1], dim=1))   # 3,         128x128

        return out


class PatchGANDiscriminator(nn.Module):
    """
    Classifies local patches of (photo, sketch) pairs as real/fake,
    conditioned on the style embedding (broadcast and concatenated
    with the input channels).
    """
    def __init__(self, base_ch=64, n_styles=3, style_dim=16):
        super().__init__()
        self.style_embedding = nn.Embedding(n_styles, style_dim)

        # Input: photo (3ch) + sketch (3ch) + style_map (style_dim ch)
        in_ch = 3 + 3 + style_dim

        self.model = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True),            # 128->64
            nn.Conv2d(base_ch, base_ch * 2, 4, 2, 1), nn.BatchNorm2d(base_ch * 2), nn.LeakyReLU(0.2, inplace=True),  # 64->32
            nn.Conv2d(base_ch * 2, base_ch * 4, 4, 2, 1), nn.BatchNorm2d(base_ch * 4), nn.LeakyReLU(0.2, inplace=True),  # 32->16
            nn.Conv2d(base_ch * 4, base_ch * 8, 4, 1, 1), nn.BatchNorm2d(base_ch * 8), nn.LeakyReLU(0.2, inplace=True),  # 16->15
            nn.Conv2d(base_ch * 8, 1, 4, 1, 1),  # -> patch-level real/fake logits, ~14x14
        )

    def forward(self, photo, sketch, style):
        style_vec = self.style_embedding(style)
        style_map = style_vec[:, :, None, None].expand(-1, -1, photo.size(2), photo.size(3))
        x = torch.cat([photo, sketch, style_map], dim=1)
        return self.model(x)  # raw logits, apply BCEWithLogitsLoss outside
