from gpt2 import GPT2
from dataclasses import dataclass
import torch.nn.functional as F
import torch
import tiktoken


@dataclass
class GPTConfig:
    vocab_size: int = 50257
    n_embed: int = 768
    max_seq_length: int = 1024
    n_layer: int = 12
    n_heads: int = 12


num_return_sequences = 5
max_length = 50

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

prefix = "Hello, I'm a language model,"

enc = tiktoken.get_encoding("gpt2")
tokens = enc.encode(prefix)
tokens = torch.tensor(tokens, dtype=torch.long)
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)  # (5, 8)
print(tokens.shape)
x = tokens.to(device)  # (B, T)

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

for i in range(x.shape[0]):
    dec_tokens = list(x[i, :max_length])
    print(enc.decode(dec_tokens))
