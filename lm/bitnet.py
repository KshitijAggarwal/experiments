# Simple BitNet implementation
# https://arxiv.org/pdf/2310.11453 1 Bit
# https://arxiv.org/abs/2402.17764 1.58 Bits
# https://github.com/microsoft/unilm/blob/master/bitnet/The-Era-of-1-bit-LLMs__Training_Tips_Code_FAQ.pdf

from torch import nn
import torch
from dataclasses import dataclass
import torch.nn.functional as F
import inspect


def binarize_weights(weights):
    """
    Quantizes the weights by binarizing them.
    Subtracts the mean. Binarizes the weights: 1 if weight > 0, -1 otherwise.
    Scales it back by the mean of abs(weights).

    Note:
        This is different from the implementation in Section 4 of Training Tips paper.
        This uses sign, which should include zeros, and would not be binary.

    Args:
        weights: The input weights to be quantized.

    Returns:
        Tensor: The quantized weights.
    """
    weights_norm = weights - torch.mean(weights)
    # binarize the weights
    beta = torch.mean(torch.abs(weights))
    w_hat = torch.where(weights_norm > 0, torch.tensor(1.0), torch.tensor(-1.0)) * beta
    return w_hat


def absmax_quant(x):
    gamma = torch.max(torch.abs(x), dim=-1, keepdim=True).values

    q_b = 2**7  # 8 bits
    err = 1e-6
    x_hat = torch.clamp(x * q_b / gamma, -q_b + err, q_b - err)
    return x_hat * gamma / q_b


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


def LayerNorm(x, dim, eps=1e-6):
    mean = x.mean(dim=dim, keepdim=True)
    var = x.var(dim=dim, keepdim=True, unbiased=False)
    return (x - mean) / torch.sqrt(var + eps)


class BitLinear(nn.Linear):
    # BitLinear with binary/ternary weights
    # Binary BitLinear: https://arxiv.org/abs/2310.11453
    # Ternary BitLinear: https://arxiv.org/pdf/2402.17764
    def forward(self, x):
        x = LayerNorm(x, -1)
        x_hat = x + (absmax_quant(x) - x).detach()
        w = self.weight

        # w_hat = w + (binarize_weights(w) - w).detach()
        w_hat = w + (ternarize_weights(w) - w).detach()
        return F.linear(x_hat, w_hat)


class MLP(nn.Module):
    # Same as MLP in gpt2 but with
    # nn.Linear replaced as BitLinear
    def __init__(self, config):
        # Linear size is n_embed x 4 * n_embed
        super().__init__()
        self.c_fc = BitLinear(
            config.n_embed, 4 * config.n_embed, bias=False
        )  # small to big
        self.gelu = nn.GELU()
        self.c_proj = BitLinear(
            4 * config.n_embed, config.n_embed, bias=False
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
        self.c_attn = BitLinear(config.n_embed, 3 * config.n_embed, bias=False)
        # self.heads = nn.ModuleList([Head(config) for _ in range(config.n_heads)])
        # Remember: n_embed = head_size * n_heads
        self.c_proj = BitLinear(config.n_embed, config.n_embed, bias=False)

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

        # I think this and te and pe will remain same as gpt.
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

    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        # params thats are 2D will be weight decayed otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]

        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]

        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)

        print(
            f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
        )
        print(
            f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
        )

        # Create AdamW optimizer and use the fused version if it is available
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        return torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused
        )


if __name__ == "__main__":

    @dataclass
    class GPTConfig:
        vocab_size: int = 50257
        n_embed: int = 768
        max_seq_length: int = 1024
        n_layer: int = 12
        n_heads: int = 12

    config = GPTConfig(
        n_layer=1,
    )

    device = "mps"
    model = BitNet(config)
    model.to(device)

    from lm.utils import gen_text
    from misc.utils import print_model_summary

    gen_text(
        model,
        num_return_sequences=5,
        max_length=25,
        device=device,
        prefix="Hello, I'm a language model,",
    )

    print_model_summary(model)
