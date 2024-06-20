# %%
# Let's try to do a very simple implementation of ViT from scratch
# from https://openreview.net/pdf?id=YicbFdNTTy

# %%

from torch import nn
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from misc.utils import print_model_summary


class MLP(nn.Module):
    # The MLP contains two layers with a GELU non-linearity.
    def __init__(self, n_embed):
        super().__init__()
        self.ff_1 = nn.Linear(n_embed, 4 * n_embed, bias=True)
        self.ff_2 = nn.Linear(4 * n_embed, n_embed, bias=True)
        self.gelu = F.gelu

    def forward(self, x):
        x = self.ff_1(x)
        x = self.gelu(x)
        x = self.ff_2(x)
        x = self.gelu(x)
        return x


class MultiHeadAttention(nn.Module):
    # Appendix A of https://openreview.net/pdf?id=YicbFdNTTy
    def __init__(self, config):
        super().__init__()
        assert config.n_embed % config.n_heads == 0
        self.qkv = nn.Linear(config.n_embed, 3 * config.n_embed, bias=True)  # Bias?
        self.n_heads = config.n_heads
        self.ff = nn.Linear(config.n_embed, config.n_embed, bias=True)  # Bias?

    def forward(self, x):
        B, N, n_embed = x.shape  # B: Batch size, N: Number of patches, D: n_embed

        qkv = self.qkv(x)  # B, N, 3 * n_embed
        q, k, v = qkv.split(n_embed, -1)  # B, N, n_embed
        q = q.view(B, N, self.n_heads, n_embed // self.n_heads).transpose(
            1, 2
        )  # (B, N, nh, n_embed / nh) -> (B, nh, N, n_embed / nh)
        k = k.view(B, N, self.n_heads, n_embed // self.n_heads).transpose(
            1, 2
        )  # (B, N, nh, n_embed / nh) -> (B, nh, N, n_embed / nh)
        v = v.view(B, N, self.n_heads, n_embed // self.n_heads).transpose(
            1, 2
        )  # (B, N, nh, n_embed / nh) -> (B, nh, N, n_embed / nh)

        attn = (
            q @ k.transpose(-1, -2) * k.size(-1) ** (-0.5)
        )  # (B, nh, N, n_embed / nh) x (B, nh, n_embed / nh, N) -> (B, nh, N, N)

        # No causal self-attention mask here. No concept of future or past tokens here. It's
        # all just tokens.

        y = (
            attn @ v
        )  # (B, nh, N, N) x (B, nh, N, n_embed / nh) -> (B, nh, N, n_embed / nh)

        y = (
            y.transpose(1, 2).contiguous().view(B, N, n_embed)
        )  # (B, N, nh, n_embed / nh) -> (B, N, nh * n_embed)
        y = self.ff(y)
        return y


class EncoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embed)
        self.mha = MultiHeadAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embed)
        self.mlp = MLP(config.n_embed)

    def forward(self, x):
        # Norm -> MHA -> Norm -> MLP
        x = x + self.mha(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class ViT(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.img_size % config.patch_size == 0

        self.patch_projection = nn.Linear(
            config.channels * config.patch_size**2, config.n_embed
        )
        self.encoder = nn.ModuleList(
            [EncoderBlock(config) for _ in range(config.n_layer)]
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, config.n_embed))

        self.n_patches = config.img_size**2 // config.patch_size**2
        self.pe = nn.Parameter(torch.randn(1, self.n_patches + 1, config.n_embed))

        self.ln = nn.LayerNorm(config.n_embed)
        self.mlp_head = nn.Linear(config.n_embed, config.num_classes)
        self.config = config

    def forward(self, x):
        # reshape x
        B, H, W, C = x.shape

        P = self.config.patch_size

        # assert (H * W) % (P * P) == 0 # not needed. Assert added in init.
        # n_patches = H * W // (P * P)

        # flatten patches
        x = x.view(B, self.n_patches, P * P * C)

        # embed patches
        x = self.patch_projection(x)  # B, n_patches, n_embed

        # add class token to patches to get B, n_patches + 1, n_embed
        cls_token = self.cls_token.expand(B, -1, -1)  # shape: B, 1, n_embed
        x = torch.cat((cls_token, x), dim=1)  # B, n_patches + 1, n_embed

        # positional encoding
        x = x + self.pe

        # transformer
        for e in self.encoder:
            x = e(x)

        # Layer norm
        x = self.ln(x)  # B, n_patches, n_embed

        # classification head?
        x = x[:, 0, :]  # B, 1, n_embed

        return self.mlp_head(x)


# %%


@dataclass
class ViTConfig:
    img_size: int = 256
    channels: int = 3
    patch_size: int = 16
    n_embed: int = 96
    n_layer: int = 8
    n_heads: int = 6
    num_classes: int = 2


B = 2

config = ViTConfig()
vitb_config = ViTConfig(patch_size=16, n_embed=768, n_heads=12, n_layer=12)

torch.manual_seed(42)

images = torch.randn(
    B, vitb_config.img_size, vitb_config.img_size, vitb_config.channels
)

vit = ViT(vitb_config)
out = vit(images)

print(out.shape)
print_model_summary(vit)

# %%
