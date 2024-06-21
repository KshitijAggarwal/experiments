# %%
import tiktoken

tinyskp = "/Users/kshitijaggarwal/Documents/Projects/experiments/data/input.txt"
with open(tinyskp, "r") as f:
    text = f.read()

data = text[:1000]

# %%

enc = tiktoken.get_encoding("gpt2")
tokens = enc.encode(data)
print(len(data), len(tokens))

# %%
# data for training
import torch

torch.set_float32_matmul_precision("high")

buf = torch.tensor(tokens[: 24 + 1], dtype=torch.long)
x = buf[:-1].view(4, 6)
y = buf[1:].view(4, 6)

# %%

from lm.gpt2 import GPT2, GPTConfig
import torch.nn as nn

model = GPT2(GPTConfig(vocab_size=50304))
ce = nn.CrossEntropyLoss()  # loss(input, target)
device = "mps"

B, T = 4, 32
buf = torch.tensor(tokens[: B * T + 1], dtype=torch.long)
buf = buf.to(device)
x = buf[:-1].view(B, T)
y = buf[1:].view(B, T)

model.to(device)
# model = torch.compile(model) # RuntimeError: Dynamo is not supported on Python 3.12+

# %%

# logits = model(x.to(device))
# # shapes: (4, 32, vocab_size) -> (4*32, vocab_size),  (4, 32) -> (4*32)
# loss = ce(logits.view(-1, logits.shape[-1]), y.view(-1))
# >> tensor(10.9285, grad_fn=<NllLossBackward0>)

# at init, every token should have a uniform prob.
# So cross entropy loss would be: -1 * log(1/vocab_size) = 10.8249

# %%
# training loop
# nsteps = 50
# optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

# model.train()
# for i in range(nsteps):
#     # Zero the gradients
#     optimizer.zero_grad()

#     # Forward Pass
#     logits = model(x.to(device))

#     # Calculate loss
#     loss = ce(logits.view(-1, logits.shape[-1]), y.view(-1))

#     # backward pass, calculate gradients
#     loss.backward()

#     # update weights
#     optimizer.step()

#     print(f"step: {i}, loss: {loss.item()}")

# step: 0, loss: 10.966878890991211
# step: 49, loss: 0.010370781645178795

# %%

# dataloader

import tiktoken
import math


class DataLoaderLite:
    def __init__(self, B, T):
        self.B = B
        self.T = T

        tinyskp = "/Users/kshitijaggarwal/Documents/Projects/experiments/data/input.txt"
        with open(tinyskp, "r") as f:
            text = f.read()

        enc = tiktoken.get_encoding("gpt2")
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens, dtype=torch.long)
        print(f"Loaded {len(self.tokens)} tokens")
        print(f"1 Epoch will have {len(self.tokens) // (B * T)} batches")

        self.current_pos = 0

    def next_batch(self):
        B, T = self.B, self.T
        start = self.current_pos
        end = self.current_pos + B * T + 1
        buf = self.tokens[start:end]
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)

        self.current_pos += B * T
        if self.current_pos + B * T + 1 > len(self.tokens):
            self.current_pos = 0
        return x, y


# %%


def get_lr(step, warmup_steps, max_steps, max_lr, min_lr):
    # warmup + cosine decay LR schedule
    # 1) Linear warmup
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps

    # 2) if step > max_steps, return min_lr
    if step > max_steps:
        return min_lr

    # 3) cosine decay in between the above two
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= progress <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + coeff * (max_lr - min_lr)


# %%

import time

B, T = 16, 64
max_lr = 6e-4  # from GPT-3 paper
min_lr = max_lr * 0.1
warmup_steps = 10
max_steps = 50
train_loader = DataLoaderLite(B, T)

# training loop
max_steps = 50
# optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8)
optimizer = model.configure_optimizers(
    weight_decay=0.1, learning_rate=6e-4, device_type=device
)

model.train()
for step in range(max_steps):
    t0 = time.time()
    x, y = train_loader.next_batch()
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad()
    # with torch.autocast(device_type=device, dtype=torch.bfloat16):
    logits = model(x.to(device))
    loss = ce(logits.view(-1, logits.shape[-1]), y.view(-1))
    loss.backward()

    # gradient clipping, GPT-3 paper.
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    # determine and set the learning rate for this iteration
    lr = get_lr(step, warmup_steps, max_steps, max_lr, min_lr)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    optimizer.step()
    t1 = time.time()
    dt = t1 - t0
    tokens_per_sec = train_loader.B * train_loader.T / dt
    print(
        f"step {step:2d}, loss: {loss.item():.6f}, lr: {lr:.4e}, norm: {norm:.4f}, dt: {dt*1000:.2f}ms, tok/sec: {tokens_per_sec:.2f}"
    )

# %%


# %%
