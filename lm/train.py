import torch
import math
from lm.gpt2 import GPT2, GPTConfig
from lm.data import DataLoaderLite
from lm.utils import get_lr, ddp_setup
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import destroy_process_group
import torch.distributed as dist

import time

torch.set_float32_matmul_precision("high")

# DDP Setup
ddp, device, ddp_rank, ddp_local_rank, ddp_world_size, master_process = ddp_setup()

# all processes get the same seed
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)


model = GPT2(GPTConfig(vocab_size=50304))
model.to(device)

if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model


ce = nn.CrossEntropyLoss()

total_batch_size = 2**14  # 2**19  # 0.5M
B, T = 16, 64
assert total_batch_size % (B * T * ddp_world_size) == 0
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)

if master_process:
    print(f"Total desired batch size: {total_batch_size}")
    print(f"Gradient accumulation steps: {grad_accum_steps}")

max_lr = 6e-4  # from GPT-3 paper
min_lr = max_lr * 0.1
warmup_steps = 10
max_steps = 50
train_loader = DataLoaderLite(
    B=B, T=T, process_rank=ddp_local_rank, num_processes=ddp_world_size
)

# training loop
max_steps = 50
# optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8)
optimizer = raw_model.configure_optimizers(
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
        if ddp:
            # we want gradients to be synced only at the last micro_step
            # not after each micro_step.
            model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
        loss.backward()

    if ddp:
        # average loss_accum across all processes and
        # modify loss_accum to that average for all processes
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

    # gradient clipping, GPT-3 paper.
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    # determine and set the learning rate for this iteration
    lr = get_lr(step, warmup_steps, max_steps, max_lr, min_lr)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    optimizer.step()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    t1 = time.time()
    dt = t1 - t0
    tokens_per_sec = (
        train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size / dt
    )
    print(
        f"step {step:2d}, loss: {loss_accum.item():.6f}, lr: {lr:.4e}, norm: {norm:.4f}, dt: {dt*1000:.2f}ms, tok/sec: {tokens_per_sec:.2f}"
    )

if ddp:
    destroy_process_group()

# torchrun --nproc_per_node=2 train.py
