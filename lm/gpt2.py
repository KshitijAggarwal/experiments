# %%

import torch.nn as nn
from dataclasses import dataclass
import torch.nn.functional as F
import torch

# transformer.h.0.ln_1.weight torch.Size([768])
# transformer.h.0.ln_1.bias torch.Size([768])
# transformer.h.0.attn.c_attn.weight torch.Size([768, 2304])
# transformer.h.0.attn.c_attn.bias torch.Size([2304])
# transformer.h.0.attn.c_proj.weight torch.Size([768, 768])
# transformer.h.0.attn.c_proj.bias torch.Size([768])
# transformer.h.0.ln_2.weight torch.Size([768])
# transformer.h.0.ln_2.bias torch.Size([768])
# transformer.h.0.mlp.c_fc.weight torch.Size([768, 3072])
# transformer.h.0.mlp.c_fc.bias torch.Size([3072])
# transformer.h.0.mlp.c_proj.weight torch.Size([3072, 768])
# transformer.h.0.mlp.c_proj.bias torch.Size([768])


@dataclass
class GPTConfig:
    vocab_size: int = 10
    n_embed: int = 100
    max_seq_length: int = 100
    n_layer: int = 8
    n_heads: int = 6


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

# %%


@dataclass
class GPTConfig:
    vocab_size: int = 50257
    n_embed: int = 768
    max_seq_length: int = 1024
    n_layer: int = 12
    n_heads: int = 12


num_return_sequences = 5
max_length = 30

device = "cpu"

if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(f"Using device: {device}")

# gptconfig = GPTConfig()
# model = GPT2(gptconfig)
model = GPT2.from_pretrained("gpt2")
model.eval()
model.to(device)

# %%

prefix = "Hello, I'm a language model,"

import tiktoken
import torch

enc = tiktoken.get_encoding("gpt2")
tokens = enc.encode(prefix)
tokens = torch.tensor(tokens, dtype=torch.long)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)  # (5, 8)
print(tokens.shape)
x = tokens.to(device)  # (B, T)

# %%

torch.manual_seed(42)
torch.cuda.manual_seed(42)
while x.shape[1] < max_length:
    with torch.no_grad():
        logits = model(x)  # B, T, vocab_size

        # take the logits of the last token
        logits = logits[:, -1, :]  # B, vocab_size

        # softmax to get probabillities
        probs = F.softmax(logits, -1)  # softmax over vocab size # B, vocab_size

        # sample top k from probs
        k = 50
        topk_probs, topk_indices = torch.topk(probs, k=k, dim=-1)  # B, k

        # sample from probs
        ix = torch.multinomial(topk_probs, 1)  # indices. Shape: B, 1

        xcol = torch.gather(topk_indices, -1, ix)

        x = torch.cat([x, xcol], dim=1)

# %%

for i in range(x.shape[0]):
    dec_tokens = list(x[i, :max_length])
    print(enc.decode(dec_tokens))

# %%
