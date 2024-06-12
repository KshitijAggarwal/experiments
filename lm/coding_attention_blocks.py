# %%

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

# %%


class Head(nn.Module):
    def __init__(self, config):
        super().__init__()
        head_size = config.n_embed // config.n_heads
        self.query = nn.Linear(config.n_embed, head_size, bias=False)
        self.key = nn.Linear(config.n_embed, head_size, bias=False)
        self.value = nn.Linear(config.n_embed, head_size, bias=False)
        # we used config.max_seq_length here as tokens will always be within that length
        self.register_buffer(
            "tril", torch.tril(torch.ones(config.max_seq_length, config.max_seq_length))
        )

    def forward(self, x):
        B, T, n_embed = x.shape
        q = self.query(x)  # B, T, head_size
        k = self.key(x)  # B, T, head_size

        # wei = q @ k.transpose(1, 2) * n_embed ** (-0.5)
        wei = (
            q @ k.transpose(-2, -1) * n_embed ** (-0.5)
        )  # (B, T, head_size) x (B, head_size, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)  # softmax to get weights for averaging along time

        v = self.value(x)
        out = wei @ v
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embed // config.n_heads
        self.heads = nn.ModuleList([Head(config) for _ in range(config.n_heads)])
        # Remember: n_embed = head_size * n_heads
        self.c_proj = nn.Linear(config.n_embed, config.n_embed, bias=True)

    def forward(self, x):
        x = torch.cat([h(x) for h in self.heads], dim=-1)
        x = self.c_proj(x)
        return x


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embed // config.n_heads
        self.config = config
        self.c_attn = nn.Linear(config.n_embed, 3 * config.n_embed)
        # self.heads = nn.ModuleList([Head(config) for _ in range(config.n_heads)])
        # Remember: n_embed = head_size * n_heads
        self.c_proj = nn.Linear(config.n_embed, config.n_embed, bias=True)
        self.register_buffer(
            "tril",
            torch.tril(
                torch.ones(self.config.max_seq_length, self.config.max_seq_length)
            ).view((1, 1, self.config.max_seq_length, self.config.max_seq_length)),
        )

    def forward(self, x):
        B, T, n_embed = x.shape
        n_heads = self.config.n_heads
        qkv = self.c_attn(x)  # B, T, 3 * n_embed
        q, k, v = qkv.split(n_embed, dim=-1)  # 3 x (B, T, n_embed)

        q = k.view(B, T, n_heads, n_embed // n_heads).transpose(
            1, 2
        )  # B, T, nh, head_size -> B, nh, T, head_size
        k = k.view(B, T, n_heads, n_embed // n_heads).transpose(
            1, 2
        )  # B, T, nh, head_size -> B, nh, T, head_size
        v = v.view(B, T, n_heads, n_embed // n_heads).transpose(
            1, 2
        )  # B, T, nh, head_size -> B, nh, T, head_size

        attn = (
            q @ k.transpose(-1, -2) * n_embed ** (-0.5)
        )  # (B, nh, T, head_size) x (B, nh, head_size, T) -> (B, nh, T, T)

        attn = attn.masked_fill(
            self.tril[:, :, :T, :T] == 0, float("-inf")
        )  # (B, nh, T, T)
        attn = F.softmax(attn, dim=-1)

        out = attn @ v  # (B, nh, T, T) x (B, nh, T, head_size) -> (B, nh, T, head_size)
        out = (
            out.transpose(1, 2).contiguous().view((B, T, n_embed))
        )  # (B, nh, T, head_size) -> (B, T, nh, head_size) -> (B, T, nh * head_size)

        out = self.c_proj(out)
        return out


# %%


@dataclass
class GPTConfig:
    vocab_size: int = 10
    n_embed: int = 96
    max_seq_length: int = 100
    n_layer: int = 8
    n_heads: int = 6


config = GPTConfig()

torch.manual_seed(42)
x = torch.randn(4, 8, config.n_embed)
B, T, n_embed = x.size()

# %%

h = Head(config)
mha = MultiHeadAttention(config)
csa = CausalSelfAttention(config)

x_h = h(x)
x_mha = mha(x)
x_csa = csa(x)

# %%

c_attn = nn.Linear(config.n_embed, 3 * config.n_embed)
c_proj = nn.Linear(config.n_embed, config.n_embed, bias=True)

tril = torch.tril(torch.ones(config.max_seq_length, config.max_seq_length)).view(
    (1, 1, config.max_seq_length, config.max_seq_length)
)

# %%

B, T, n_embed = x.shape
n_heads = config.n_heads
qkv = c_attn(x)  # B, T, 3 * n_embed
q, k, v = qkv.split(config.n_embed, dim=-1)  # 3 x (B, T, n_embed)

q = k.view(B, T, n_heads, n_embed // n_heads).transpose(
    1, 2
)  # B, T, nh, head_size -> B, nh, T, head_size
k = k.view(B, T, n_heads, n_embed // n_heads).transpose(
    1, 2
)  # B, T, nh, head_size -> B, nh, T, head_size
v = v.view(B, T, n_heads, n_embed // n_heads).transpose(
    1, 2
)  # B, T, nh, head_size -> B, nh, T, head_size

attn = (
    q @ k.transpose(-1, -2) * n_embed ** (-0.5)
)  # (B, nh, T, head_size) x (B, nh, head_size, T) -> (B, nh, T, T)

# %%

attn = attn.masked_fill(tril[:, :, :T, :T] == 0, float("-inf"))  # (B, nh, T, T)
attn = F.softmax(attn, dim=-1)

out = attn @ v  # (B, nh, T, T) x (B, nh, T, head_size) -> (B, nh, T, head_size)
out = (
    out.transpose(1, 2).contiguous().view((B, T, n_embed))
)  # (B, nh, T, head_size) -> (B, T, nh, head_size) -> (B, T, nh * head_size)

out = c_proj(out)
# return x


# %%

# %%
