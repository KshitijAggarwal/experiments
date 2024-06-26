import torch
import math
from lm.gpt2 import GPT2
from lm.bitnet import BitNet
from lm.utils import GPTConfig
from lm.data import DataLoaderLite
from lm.utils import get_lr, ddp_setup, gen_text
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import destroy_process_group
import torch.distributed as dist

import time
import glob
import wandb
import os

torch.set_float32_matmul_precision("high")

#########
# Initial setup
#########

model_config = GPTConfig(vocab_size=50304, n_layer=1)
model_class = GPT2  # BitNet
total_batch_size = 2**14  # 2**19 is 0.5M
B = 16  # 8
T = 32  # 1024
max_lr = 2.4e-3  # 6e-4  # from GPT-3 paper
min_lr = max_lr * 0.1
warmup_steps = 2  # 715  # 100 is good enough: Karpathy
max_steps = 10
wandb_log = False
train_files = [
    "/Users/kshitijaggarwal/Documents/Projects/experiments/data/sample-1MT.npy"
]
val_files = train_files

# train_files = glob.glob("/workspace/edu_fineweb10B/*train*")
# val_files = glob.glob("/workspace/edu_fineweb10B/*val*")

weight_decay = 0.1
val_every_n_steps = 5
use_compile = False
#########
# End setup
#########

log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"log.txt")
with open(log_file, "w") as f:  # open for writing to clear the file
    pass

# DDP Setup
ddp, device, ddp_rank, ddp_local_rank, ddp_world_size, master_process = ddp_setup()

if "cuda" in device:
    device_type = "cuda"
else:
    device_type = device

# all processes get the same seed
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

model = model_class(model_config)
model.to(device)
if torch.cuda.is_available() and use_compile:
    model = torch.compile(model)
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model

ce = nn.CrossEntropyLoss()

assert total_batch_size % (B * T * ddp_world_size) == 0
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)

if master_process:
    print(f"Total desired batch size: {total_batch_size}")
    print(f"Gradient accumulation steps: {grad_accum_steps}")

train_loader = DataLoaderLite(
    B=B,
    T=T,
    process_rank=ddp_local_rank,
    num_processes=ddp_world_size,
    tokenfiles=train_files,
)

val_dataloader = DataLoaderLite(
    B=B,
    T=T,
    process_rank=ddp_local_rank,
    num_processes=ddp_world_size,
    tokenfiles=val_files,
)

# training loop
# this learning_rate value doesn't matter as we are using scheduler and
# lr will be obtained from scheduler during training
optimizer = raw_model.configure_optimizers(
    weight_decay=weight_decay, learning_rate=6e-4, device_type=device_type
)

import torch
import os


def save_checkpoint(
    model, model_config, optimizer, loss, step, filename="checkpoint.pth", **kwargs
):
    checkpoint = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        # Learning rate scheduler
        "scheduler_state_dict": (
            kwargs.get("scheduler").state_dict() if kwargs.get("scheduler") else None
        ),
        # Training configuration
        "train_config": kwargs.get("train_config", {}),
        # Validation metrics
        "val_loss": kwargs.get("val_loss"),
        # Dataset information
        "dataset_info": kwargs.get("dataset_info", {}),
        # Model architecture
        "model_name": model.__class__.__name__,
        "model_config": model_config,
        # Random states for reproducibility
        "torch_rng_state": torch.get_rng_state(),
    }

    torch.save(checkpoint, filename)
    print(f"Checkpoint saved to {filename}")


def common_step(model, dataloader, device, accum_steps, loss_fn, mode="train"):
    if "cuda" in device:
        device_type = "cuda"
    else:
        device_type = device

    loss_accum = 0.0
    for micro_step in range(accum_steps):
        x, y = dataloader.next_batch()
        x, y = x.to(device), y.to(device)
        if torch.cuda.is_available():
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                logits = model(x.to(device))
                loss = loss_fn(logits.view(-1, logits.shape[-1]), y.view(-1))
        else:
            logits = model(x.to(device))
            loss = loss_fn(logits.view(-1, logits.shape[-1]), y.view(-1))
        loss = loss / accum_steps
        loss_accum += loss.detach()  # detaching tensor from the graph
        if mode == "train":
            if ddp:
                # we want gradients to be synced only at the last micro_step
                # not after each micro_step.
                model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
            loss.backward()

    if ddp:
        # average loss_accum across all processes and
        # modify loss_accum to that average for all processes
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
    return loss_accum, model


# Initialize wandb
# if master_process and wandb_log:
#     wandb.init(
#         project="lm-training",
#         dir="/Users/kshitijaggarwal/Documents/Projects/experiments/",
#         config={
#             "learning_rate": 6e-4,
#             "architecture": "GPT2",
#             "dataset": "edu_fineweb10B",
#             "epochs": 19073,
#         },
#     )

model.train()
for step in range(max_steps):
    t0 = time.time()
    last_step = step == max_steps - 1

    if (step > 0 and step % val_every_n_steps == 0) or last_step:
        val_accum_steps = grad_accum_steps
        model.eval()
        val_dataloader.reset()
        with torch.no_grad():
            val_loss_accum, model = common_step(
                model, val_dataloader, device, val_accum_steps, ce, mode="val"
            )
        if master_process:
            print(f"Validation loss: {val_loss_accum.item():.4f}")
            with open(log_file, "a") as f:
                f.write(f"{step} val {val_loss_accum.item():.4f}\n")

            if step % 10 * val_every_n_steps or last_step:
                checkpoint_path = os.path.join(
                    log_dir, f"model_{raw_model.__class__.__name__}_{step:05d}.pt"
                )

                save_checkpoint(
                    model=raw_model,
                    model_config=model_config.__dict__,
                    optimizer=optimizer,
                    loss=loss_accum.item(),
                    step=step,
                    val_loss=val_loss_accum.item(),
                    filename=checkpoint_path,
                )
            # if wandb_log:
            #     wandb.log({"val_loss": val_loss_accum.item()}, step=step)

    if ((step > 0 and step % val_every_n_steps == 0) or last_step) and (
        not use_compile
    ):
        if master_process:
            max_length = 32
            gen_text(model, 1, max_length, device)

    model.train()
    optimizer.zero_grad()
    loss_accum, model = common_step(
        model, train_loader, device, grad_accum_steps, ce, mode="train"
    )

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
        with open(log_file, "a") as f:
            f.write(f"{step} train {loss_accum.item():.6f}\n")

        # if wandb_log:
        #     wandb.log(
        #         {
        #             "train_loss": loss_accum.item(),
        #             "learning_rate": lr,
        #             "grad_norm": norm,
        #         },
        #         step=step,
        #     )

if ddp:
    destroy_process_group()

if master_process and wandb_log:
    wandb.finish()

# torchrun --nproc_per_node=1 train.py
