import torch
import math
from lm.gpt2 import GPT2, GPTConfig
from lm.data import DataLoaderLite
from lm.utils import get_lr
import torch.nn as nn
import time

torch.set_float32_matmul_precision("high")

model = GPT2(GPTConfig(vocab_size=50304))
device = "mps"
model.to(device)

ce = nn.CrossEntropyLoss()

total_batch_size = 2**14  # 2**19  # 0.5M
B, T = 16, 64
assert total_batch_size % (B * T) == 0
grad_accum_steps = total_batch_size // (B * T)

print(f"Total desired batch size: {total_batch_size}")
print(f"Gradient accumulation steps: {grad_accum_steps}")

max_lr = 6e-4  # from GPT-3 paper
min_lr = max_lr * 0.1
warmup_steps = 10
max_steps = 50
train_loader = DataLoaderLite(B=B, T=T)

# training loop
max_steps = 50
# optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8)
optimizer = model.configure_optimizers(
    weight_decay=0.1, learning_rate=6e-4, device_type=device
)

model.train()
for step in range(max_steps):
    t0 = time.time()
    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)

        # with torch.autocast(device_type=device, dtype=torch.bfloat16):
        logits = model(x.to(device))
        loss = ce(logits.view(-1, logits.shape[-1]), y.view(-1))
        loss = (
            loss / grad_accum_steps
        )  # because otherwise it is a sum of losses, not average, as we are accumulating steps
        loss_accum += loss.detach()  # detaching tensor from the graph
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
    tokens_per_sec = train_loader.B * train_loader.T * grad_accum_steps / dt
    print(
        f"step {step:2d}, loss: {loss_accum.item():.6f}, lr: {lr:.4e}, norm: {norm:.4f}, dt: {dt*1000:.2f}ms, tok/sec: {tokens_per_sec:.2f}"
    )
