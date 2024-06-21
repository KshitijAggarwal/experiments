import torch.nn as nn
from dataclasses import dataclass
import torch.nn.functional as F
import torch
import inspect


@dataclass
class GPTConfig:
    vocab_size: int = 50257
    n_embed: int = 768
    max_seq_length: int = 1024
    n_layer: int = 12
    n_heads: int = 12


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embed % config.n_heads == 0
        self.config = config
        self.c_attn = nn.Linear(config.n_embed, 3 * config.n_embed, bias=True)
        # self.heads = nn.ModuleList([Head(config) for _ in range(config.n_heads)])
        # Remember: n_embed = head_size * n_heads
        self.c_proj = nn.Linear(config.n_embed, config.n_embed, bias=True)

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


class MLP(nn.Module):
    def __init__(self, config):
        # Linear size is n_embed x 4 * n_embed
        super().__init__()
        self.c_fc = nn.Linear(
            config.n_embed, 4 * config.n_embed, bias=True
        )  # small to big
        # GPT 2 used 'tanh' approx, but there is no speed advantage anymore of using it.
        # So can be skipped.
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(
            4 * config.n_embed, config.n_embed, bias=True
        )  # big to small
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embed)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embed)
        self.mlp = MLP(config)

    def forward(self, x):
        # note both ln are before attention or MLP. The residual stream is clear.
        # and free from any LN
        # MLP is also referred to as FFN
        # attn is an aggregation, pooling, weighted sum function
        # MLP is independent done on all tokens. So it is a map function
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT2(nn.Module):
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
        self.lm_head = nn.Linear(
            config.n_embed, config.vocab_size, bias=False
        )  # from n_embed to vocab size

        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight  # copies the pointer

        # init params
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "NANOGPT_SCALE_INIT"):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            # almost similar to Xavier initialization
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    @classmethod
    def from_pretrained(cls, model_type):
        assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            "gpt2": dict(n_layer=12, n_heads=12, n_embed=768),  # 124M params
            "gpt2-medium": dict(n_layer=24, n_heads=16, n_embed=1024),  # 350M params
            "gpt2-large": dict(n_layer=36, n_heads=20, n_embed=1280),  # 774M params
            "gpt2-xl": dict(n_layer=48, n_heads=25, n_embed=1600),  # 1558M params
        }[model_type]
        config_args["vocab_size"] = 50257  # always 50257 for GPT model checkpoints
        config_args["max_seq_length"] = 1024  # always 1024 for GPT model checkpoints

        gptconfig = GPTConfig(**config_args)

        model = GPT2(gptconfig)
        sd = model.state_dict()
        sd_keys = sd.keys()

        sd_keys = [
            l for l in sd_keys if not l.endswith(".attn.bias")
        ]  # discard this mask / buffer, not a param

        from transformers import GPT2LMHeadModel

        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # from https://github.com/karpathy/build-nanogpt/blob/842135596a31f50c0095952e6e0dc2c9fa35a22a/train_gpt2.py#L160C1-L167C104
        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [
            k for k in sd_keys_hf if not k.endswith(".attn.masked_bias")
        ]  # ignore these, just a buffer
        sd_keys_hf = [
            k for k in sd_keys_hf if not k.endswith(".attn.bias")
        ]  # same, just the mask (buffer)
        transposed = [
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
        ]
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(
            sd_keys
        ), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"

        # Checking keys
        # for k in sd_keys:
        #     if k not in sd_keys_hf:
        #         print(k)
        # print('--')
        # for k in sd_keys_hf:
        #     if k not in sd_keys:
        #         print(k)

        # Let's copy weights
        for k in sd_keys_hf:
            to_transpose = any([k.endswith(l) for l in transposed])
            if to_transpose:
                assert sd[k].shape[::-1] == sd_hf[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].T)
            else:
                assert sd[k].shape == sd_hf[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

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


# config_gpt2_124M = GPTConfig(
#     vocab_size=50257,  # 50000 BPE merges, 256 bytes tokens, 1 special token
#     n_embed=768,
#     max_seq_length=1024,
#     n_layer=12,
#     n_heads=12,
# )

# model = GPT2(config_gpt2_124M)
# model = GPT2.from_pretrained("gpt2")
# # print model's state dict
# sd = model.state_dict()
# for k, v in sd.items():
#     print(k, v.shape)

# from transformers import pipeline, set_seed
# generator = pipeline("text-generation", model="gpt2")
# set_seed(42)
# generator("Hello, I'm a language model,", max_length=30, num_return_sequences=5)
