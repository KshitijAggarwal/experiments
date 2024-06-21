# %%
# import tiktoken

# tinyskp = "/Users/kshitijaggarwal/Documents/Projects/experiments/data/input.txt"
# with open(tinyskp, "r") as f:
#     text = f.read()

# data = text[:1000]

# # %%

# enc = tiktoken.get_encoding("gpt2")
# tokens = enc.encode(data)
# print(len(data), len(tokens))

# %%
# data for training
import torch
import math
import time

# buf = torch.tensor(tokens[: 24 + 1], dtype=torch.long)
# x = buf[:-1].view(4, 6)
# y = buf[1:].view(4, 6)

# %%

from lm.gpt2 import GPT2, GPTConfig
from lm.data import DataLoaderLite
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.distributed import destroy_process_group

# model = GPT2(GPTConfig(vocab_size=50304))
# ce = nn.CrossEntropyLoss()  # loss(input, target)
# device = "mps"
# model.to(device)

# B, T = 4, 32
# buf = torch.tensor(tokens[: B * T + 1], dtype=torch.long)
# buf = buf.to(device)
# x = buf[:-1].view(B, T)
# y = buf[1:].view(B, T)


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

# %%

from lm.utils import ddp_setup

ddp, device, ddp_rank, ddp_local_rank, ddp_world_size, master_process = ddp_setup()

# from torch.distributed import init_process_group, destroy_process_group
# import os


# from torch.nn.parallel import DistributedDataParallel as DDP
# import torch.distributed as dist

# # setup DDP (Distributed Data Parallel)
# # torchrun command sets the env variables: RANK, LOCAL_RANK, WORLD_SIZE
# # ddp = int(os.environ.get("RANK", -1)) != -1
# ddp = int(os.environ.get("WORLD_SIZE", "1")) > 1
# if ddp:
#     if torch.cuda.is_available():
#         backend = "nccl"  # need nccl for cuda + DDP
#     else:
#         backend = "gloo"  # for ddp on cpu
#     init_process_group(backend=backend)
#     ddp_rank = int(os.environ["RANK"])  # uid for all processes across all nodes
#     ddp_local_rank = int(os.environ["LOCAL_RANK"])  # uid for all processes on a node
#     ddp_world_size = int(os.environ["WORLD_SIZE"])  # total no. of processes running
#     if torch.cuda.is_available():
#         device = f"cuda:{ddp_local_rank}"
#         torch.cuda.set_device(device)
#     elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
#         device = "mps"
#     else:
#         device = "cpu"
#     master_process = ddp_rank == 0  # for logging, checkpointing, etc
# else:
#     # non DDP run
#     ddp_rank = 0
#     ddp_local_rank = 0
#     ddp_world_size = 1
#     master_process = True
#     device = "cpu"
#     if torch.cuda.is_available():
#         device = "cuda"
#     elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
#         device = "mps"
# print(f"Using device: {device}")


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

# all processes get the same seed
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)


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

torch.set_float32_matmul_precision("high")

model = GPT2(GPTConfig(vocab_size=50304))
ce = nn.CrossEntropyLoss()  # loss(input, target)
model.to(device)
# model = torch.compile(model) # RuntimeError: Dynamo is not supported on Python 3.12+
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

raw_model = model.module if ddp else model

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
    if master_process:
        print(
            f"step {step:2d}, loss: {loss_accum.item():.6f}, lr: {lr:.4e}, norm: {norm:.4f}, dt: {dt*1000:.2f}ms, tok/sec: {tokens_per_sec:.2f}"
        )

if ddp:
    destroy_process_group()

# %%
# When trying this on my M3 macbook pro:
# torchrun --nproc_per_node=2 training_prep.py
# NotImplementedError: The operator 'c10d::allgather_' is not currently implemented for the MPS device.
# If you want this op to be added in priority during the prototype phase of this feature, please comment
# on https://github.com/pytorch/pytorch/issues/77764. As a temporary fix, you can set the environment variable
# `PYTORCH_ENABLE_MPS_FALLBACK=1` to use the CPU as a fallback for this op. WARNING: this will be slower than running natively on MPS.


# %%
