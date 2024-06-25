# %%
# Simple BitNet implementation
# https://arxiv.org/pdf/2310.11453 1 Bit
# https://arxiv.org/abs/2402.17764 1.58 Bits
# https://github.com/microsoft/unilm/blob/master/bitnet/The-Era-of-1-bit-LLMs__Training_Tips_Code_FAQ.pdf

# %%

from torch import nn
import torch


class MLP(nn.Module):
    # Same as MLP in gpt2 but with
    # nn.Linear replaced as BitLinear
    def __init__(self, config):
        # Linear size is n_embed x 4 * n_embed
        super().__init__()
        self.c_fc = BitLinear(
            config.n_embed, 4 * config.n_embed, bias=True
        )  # small to big
        self.gelu = nn.GELU()
        self.c_proj = BitLinear(
            4 * config.n_embed, config.n_embed, bias=True
        )  # big to small

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class CausalSelfAttention(nn.Module):
    # Same as CausalSelfAttention in gpt2 but with
    # nn.Linear replaced as BitLinear
    def __init__(self, config):
        super().__init__()
        assert config.n_embed % config.n_heads == 0
        self.config = config
        self.c_attn = BitLinear(config.n_embed, 3 * config.n_embed, bias=True)
        # self.heads = nn.ModuleList([Head(config) for _ in range(config.n_heads)])
        # Remember: n_embed = head_size * n_heads
        self.c_proj = BitLinear(config.n_embed, config.n_embed, bias=True)

        # not really a 'bias', more of a mask, but following the OpenAI/HF naming though
        self.register_buffer(
            "bias",
            torch.tril(
                torch.ones(self.config.max_seq_length, self.config.max_seq_length)
            ).view((1, 1, self.config.max_seq_length, self.config.max_seq_length)),
        )

    def forward(self, x):
        B, T, n_embed = x.shape
        n_heads = self.config.n_heads
        qkv = self.c_attn(x)  # B, T, 3 * n_embed
        q, k, v = qkv.split(n_embed, dim=-1)  # 3 x (B, T, n_embed)

        q = q.view(B, T, n_heads, n_embed // n_heads).transpose(
            1, 2
        )  # B, T, nh, head_size -> B, nh, T, head_size
        k = k.view(B, T, n_heads, n_embed // n_heads).transpose(
            1, 2
        )  # B, T, nh, head_size -> B, nh, T, head_size
        v = v.view(B, T, n_heads, n_embed // n_heads).transpose(
            1, 2
        )  # B, T, nh, head_size -> B, nh, T, head_size

        # out = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # flash attention
        # B and nh are treated as batch dimension.
        attn = (
            q @ k.transpose(-1, -2) * k.size(-1) ** (-0.5)
        )  # (B, nh, T, head_size) x (B, nh, head_size, T) -> (B, nh, T, T)

        attn = attn.masked_fill(
            self.bias[:, :, :T, :T] == 0, float("-inf")
        )  # (B, nh, T, T)
        attn = F.softmax(attn, dim=-1)
        out = attn @ v  # (B, nh, T, T) x (B, nh, T, head_size) -> (B, nh, T, head_size)
        out = (
            out.transpose(1, 2).contiguous().view((B, T, n_embed))
        )  # (B, nh, T, head_size) -> (B, T, nh, head_size) -> (B, T, nh * head_size)

        out = self.c_proj(out)
        return out


class Block(nn.Module):
    # exactly same as Block in gpt2
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embed)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embed)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class BitNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(
                    self.config.vocab_size, self.config.n_embed
                ),  # token embedding
                wpe=nn.Embedding(
                    self.config.max_seq_length, self.config.n_embed
                ),  # positional embeddings
                h=nn.ModuleList([Block(config) for l in range(config.n_layer)]),
                ln_f=nn.LayerNorm(normalized_shape=config.n_embed),
            )
        )

        # I think this and te and pe will remain non-bit wise.
        self.lm_head = nn.Linear(
            config.n_embed, config.vocab_size, bias=False
        )  # from n_embed to vocab size

        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight  # copies the pointer

        # init params
        self.apply(self._init_weights)

    def _init_weights(self, module):
        pass

    def forward(self, idx):
        B, T = idx.shape
        assert (
            T <= self.config.max_seq_length
        ), f"Max sequence length allowed is {self.config.max_seq_length}"

        pe = torch.arange(0, T, dtype=torch.long, device=idx.device)  # shape (T)
        wpe = self.transformer.wpe(pe)  # position embeddings. Shape: T, n_embed
        wte = self.transformer.wte(idx)  # word token embeddings. Shape: B, T, n_embed

        x = wte + wpe  # note the hidden broadcasting happening here.

        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)  # B, T, n_embed

        logits = self.lm_head(x)  # B, T, vocab_size
        return logits


# %%

import torch
from torch import nn

# self.l = nn.Linear(in_features, out_features, bias)
in_features = 16
out_features = 32

weights = torch.randn((in_features, out_features), requires_grad=True)

# %%

# class BitLinear(nn.Module):
#     # this is the replacement of nn.Linear
#     def __init__(self, in_features, out_features, config, q_b, before_nl=False):

#         self.ln = nn.LayerNorm(config.n_embed)
#         self.linear = nn.Linear(in_features, out_features, bias=False)
#         self.q_b = q_b
#         self.before_nl = False
#         self.err = 1e-8

#     def forward(self, x):
#         x = self.ln(x)

#         # centralize the weights to zero mean
#         weights = self.linear.weight
#         weights_norm = weights - torch.mean(weights)
#         # binarize the weights
#         weights_quant = torch.sign(weights_norm).
#         self.linear.weight = weights

#         gamma = torch.max(x) # absolute max of x (per batch?)
#         if self.before_nl:
#             eta = torch.min(x)
#             x = absmax_quant((x - eta) * self.q_b / gamma, self.err, self.q_b - self.err)
#         else:
#             x = absmax_quant(x * self.q_b / gamma, -self.q_b + self.err, self.q_b - self.err)


#         # dequantization
#         beta = torch.norm(weights, p=1)
#         y = self.linear(x) * gamma * beta / self.q_b
#         return y

# %%

# existing implementation..
# from https://github.com/microsoft/unilm/blob/master/bitnet/The-Era-of-1-bit-LLMs__Training_Tips_Code_FAQ.pdf
import torch.nn.functional as F
import torch
from torch import nn


def activation_quant(x):
    """Per token quantization to 8bits. No grouping is needed for quantization

    Args:
        x (Tensor): _description_

    Returns:
        _type_: _description_
    """
    scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
    y = (x * scale).round().clamp_(-128, 127) / scale
    return y


# def weight_quant(w): # this is for binary
#     scale = w.abs().mean()
#     e = w.mean()
#     u = (w - e).sign() * scale
#     return u


def weight_quant(w):  # this is for ternary
    scale = 1.0 / w.abs().mean().clamp_(min=1e-5)
    u = (w * scale).round().clamp_(-1, 1) / scale
    return u


def LayerNorm(x, dim, eps=1e-6):
    mean = x.mean(dim=dim, keepdim=True)
    var = x.var(dim=dim, keepdim=True, unbiased=False)
    return (x - mean) / torch.sqrt(var + eps)


class BitLinear2(nn.Linear):
    # shape of x is n x d
    def forward(self, x):
        w = self.weight
        x_norm = LayerNorm(x, -1)

        # STE using detach
        x_quant = x_norm + (activation_quant(x_norm) - x_norm).detach()
        w_quant = w + (weight_quant(w) - w).detach()
        y = F.linear(x_quant, w_quant)
        return y


# %%


# My implementation
def quantize_weights(weights):
    weights_norm = weights - torch.mean(weights)
    # binarize the weights
    beta = torch.mean(torch.abs(weights))
    w_hat = (
        torch.where(weights_norm > 0, torch.tensor(1.0), torch.tensor(-1.0)) * beta
    )  # for binary
    return w_hat


def ternarize_weights(weights, eps=1e-6):
    """
    Quantizes the weights by ternarizing them (-1, 0, 1).

    Args:
        weights: The input weights to be quantized.

    Returns:
        Tensor: The quantized weights.
    """

    gamma = torch.mean(torch.abs(weights))
    weights = weights / (gamma + eps)
    weights = weights.round().sign() * gamma
    return weights


def absmax_quant(x):
    gamma = torch.max(torch.abs(x), dim=-1, keepdim=True).values

    q_b = 2**7
    err = 1e-6
    x_hat = torch.clamp(x * q_b / gamma, -q_b + err, q_b - err)
    return x_hat * gamma / q_b


class BitLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)

    def forward(self, x):
        x = LayerNorm(x, -1)
        x_hat = x + (absmax_quant(x) - x).detach()
        w = self.linear.weight
        w_hat = w + (quantize_weights(w) - w).detach()
        self.linear.weight = torch.nn.Parameter(w_hat)
        return self.linear(x_hat)


class BitLinear3(nn.Linear):
    def forward(self, x):
        x = LayerNorm(x, -1)
        x_hat = x + (absmax_quant(x) - x).detach()
        w = self.weight
        w_hat = w + (ternarize_weights(w) - w).detach()
        return F.linear(x_hat, w_hat)


# %%

B = 2
input = torch.rand(B, 16)
# model = ToyModel(16, 32)
bl3 = BitLinear3(16, 32)  # ternary weights
bl2 = BitLinear2(16, 32)  # existing implementation
bl1 = BitLinear(16, 32)  # binary weights

# %%

x1 = absmax_quant(input)
x2 = activation_quant(input)

# %%

w = torch.nn.Parameter(torch.randn([32, 16]))
bl1.linear.weight = w
bl2.weight = w
bl3.weight = w

w1 = quantize_weights(w)
w2 = weight_quant(w)
w3 = ternarize_weights(w)

# %%

o1 = bl1(input)
o2 = bl2(input)
o3 = bl3(input)

# %%


# %%


# %%
