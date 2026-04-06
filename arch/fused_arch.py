import torch
import torch.nn as nn
import torch.nn.functional as F
from .mirnet_v2_mini_arch import MIRNet_v2  # Assuming your LYT model is in model.py
from .mirnet_v2_mini_arch import MRB

# Feature extractor shared between under and over images
class FeatureExtractor(nn.Module):
    def __init__(self, in_channels=3, out_channels=64):
        super(FeatureExtractor, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.encoder(x)

# Attention-based fusion block
class AttentionFusionBlock(nn.Module):
    def __init__(self, channels):
        super(AttentionFusionBlock, self).__init__()
        self.fusion = nn.Sequential(
            nn.Conv2d(channels * 3, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 1, kernel_size=3, padding=1),
            nn.Sigmoid()  # Output spatial attention map
        )

    def forward(self, f_under, f_over):
        diff = torch.abs(f_under - f_over)
        combined = torch.cat([f_under, f_over, diff], dim=1)
        alpha = self.fusion(combined)
        fused = alpha * f_under + (1 - alpha) * f_over
        return fused

# Decoder that reconstructs enhanced RGB image
class Decoder(nn.Module):
    def __init__(self, channels):
        super(Decoder, self).__init__()
        self.decode = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            #nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 3, kernel_size=3, padding=1),
            nn.Sigmoid()  # Normalize output to [0, 1]
        )

    def forward(self, x):
        return self.decode(x)

# # Full LYTMEF model
# class MIRNetFused(nn.Module):
#     def __init__(self):
#         super(MIRNetFused, self).__init__()
#         self.lyt = MIRNet_v2()
#         feat_channels = 64
#         self.shallow_feat = nn.Conv2d(3, feat_channels, kernel_size=3, padding=1)

#         self.encoder = MRB(
#             n_feat=feat_channels,     # e.g., 64
#             height=3,                 # three scales: 1x, 2x, 4x
#             width=2,                  # fixed (not deeply used)
#             chan_factor=1.5,          # scale channels across resolution
#             bias=False,
#             groups=1
#         )

#         self.fusion = AttentionFusionBlock(channels=feat_channels)
#         self.decoder = Decoder(channels=feat_channels)

#     def forward(self, I_under):
#         I_over = self.lyt(I_under)
#         F_under = self.encoder(self.shallow_feat(I_under))
#         F_over  = self.encoder(self.shallow_feat(I_over))
#         F_fused = self.fusion(F_under, F_over)
#         I_final = self.decoder(F_fused)
#         return I_final  # Returning I_over for optional auxiliary loss

class MIRNetFused(nn.Module):
    def __init__(self):
        super(MIRNetFused, self).__init__()
        self.lyt = MIRNet_v2()
        feat_channels = 64
        self.shallow_feat = nn.Conv2d(3, feat_channels, kernel_size=3, padding=1)
        self.shallow_feat_nir = nn.Conv2d(3, feat_channels, kernel_size=3, padding=1)  # for upscaled NIR

        self.encoder = MRB(
            n_feat=feat_channels,
            height=3,
            width=2,
            chan_factor=1.5,
            bias=False,
            groups=1
        )

        self.fusion = AttentionFusionBlock(channels=feat_channels)
        self.decoder = Decoder(channels=feat_channels)

    def forward(self, rgb, nir_up):
        # 1. Generate "overexposed" RGB using MIRNet
        I_over = self.lyt(rgb)

        # 2. Extract features
        F_over = self.encoder(self.shallow_feat(I_over))
        F_nir  = self.encoder(self.shallow_feat_nir(nir_up))  # upscaled NIR

        # 3. Fuse both features
        F_fused = self.fusion(F_over, F_nir)

        # 4. Decode to final output
        I_final = self.decoder(F_fused)

        return I_final
