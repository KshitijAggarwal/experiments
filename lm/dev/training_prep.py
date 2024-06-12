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
# RuntimeError: Dynamo is not supported on Python 3.12+
# model = torch.compile(model)

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

import time

B, T = 16, 128
train_loader = DataLoaderLite(B, T)

# training loop
nsteps = 50
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

model.train()
for i in range(nsteps):
    t0 = time.time()
    x, y = train_loader.next_batch()
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad()
    # with torch.autocast(device_type=device, dtype=torch.bfloat16):
    logits = model(x.to(device))
    loss = ce(logits.view(-1, logits.shape[-1]), y.view(-1))
    loss.backward()
    optimizer.step()
    t1 = time.time()
    print(f"step: {i}, loss: {loss.item()}, dt: {t1-t0}")

# %%


# %%
