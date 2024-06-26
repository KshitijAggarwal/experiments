import math
import os
import torch
from torch.distributed import init_process_group
import os
import tiktoken
import torch.nn.functional as F
from dataclasses import dataclass


def get_lr(step, warmup_steps, max_steps, max_lr, min_lr):
    """
    Calculate the learning rate for a given step in a warmup and cosine decay learning rate schedule.

    Parameters:
        step (int): The current step in the learning rate schedule.
        warmup_steps (int): The number of steps for the linear warmup phase.
        max_steps (int): The total number of steps in the learning rate schedule.
        max_lr (float): The maximum learning rate.
        min_lr (float): The minimum learning rate.

    Returns:
        float: The learning rate for the given step.

    Raises:
        AssertionError: If the progress value is not between 0 and 1.

    The function implements a warmup and cosine decay learning rate schedule. The learning rate starts with a linear warmup phase, followed by a cosine decay phase. The warmup phase is defined by the `warmup_steps` parameter, and the cosine decay phase is defined by the `max_steps` and `max_lr` parameters. The minimum learning rate is defined by the `min_lr` parameter.

    The function calculates the progress value as the ratio of the current step to the total number of steps in the schedule. It then calculates the cosine decay coefficient using the progress value. Finally, it returns the learning rate by adding the minimum learning rate to the product of the cosine decay coefficient and the difference between the maximum learning rate and the minimum learning rate.

    Note: The function assumes that the `step` parameter is non-negative and less than or equal to the `max_steps` parameter.
    """
    # warmup + cosine decay LR schedule
    # 1) Linear warmup
    if step < warmup_steps:
        return min_lr + (max_lr - min_lr) * (step / warmup_steps)

    # 2) if step > max_steps, return min_lr
    if step >= max_steps:
        return min_lr

    # 3) cosine decay in between the above two
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= progress <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + coeff * (max_lr - min_lr)


def ddp_setup():
    # setup DDP (Distributed Data Parallel)
    # torchrun command sets the env variables: RANK, LOCAL_RANK, WORLD_SIZE
    # ddp = int(os.environ.get("RANK", -1)) != -1
    ddp = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if ddp:
        if torch.cuda.is_available():
            backend = "nccl"  # need nccl for cuda + DDP
        else:
            backend = "gloo"  # for ddp on cpu
        init_process_group(backend=backend)
        ddp_rank = int(os.environ["RANK"])  # uid for all processes across all nodes
        ddp_local_rank = int(
            os.environ["LOCAL_RANK"]
        )  # uid for all processes on a node
        ddp_world_size = int(os.environ["WORLD_SIZE"])  # total no. of processes running
        if torch.cuda.is_available():
            device = f"cuda:{ddp_local_rank}"
            torch.cuda.set_device(device)
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        master_process = ddp_rank == 0  # for logging, checkpointing, etc
    else:
        # non DDP run
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        master_process = True
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
    print(f"Using device: {device}")
    return ddp, device, ddp_rank, ddp_local_rank, ddp_world_size, master_process


def gen_text(
    model,
    num_return_sequences,
    max_length=0,
    ntokens=0,
    device="cpu",
    prefix="Hello, I'm a language model,",
    return_format="text",
    return_new=False,
):
    """
    Generate text using the given model.

    Args:
        model (torch.nn.Module): The language model to use for generating text.
        num_return_sequences (int): The number of sequences to generate.
        max_length (int): The maximum length (in tokens) of each generated sequence.
        ntokens (int): The number of tokens to generate.
        device (str): The device to use for generating text.
        prefix (str, optional): The prefix to use to start generating text. It can either be
        a string or a list of tokens. Defaults to "Hello, I'm a language model,".
        return_format (str, optional): The format to return the generated text in (could be text or tokens). It can either be
        "text" or "list". Defaults to "text".
        return_new (bool, optional): Whether to return only the predictions or the whole sequence.

    Returns:
        None

    """

    assert ntokens > 0 or max_length > 0

    if type(prefix) == str:
        enc = tiktoken.get_encoding("gpt2")
        tokens = enc.encode(prefix)
    elif type(prefix) == list:
        tokens = prefix
    else:
        raise ValueError("prefix must be either a string or a list of tokens")
    tokens = torch.tensor(tokens, dtype=torch.long)
    tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
    x = tokens.to(device)  # (B, T)
    B, T = x.shape
    if max_length > 0:
        assert T < max_length
        ntokens = max_length - len(tokens)

    if "cuda" in device:
        device_type = "cuda"
    else:
        device_type = device

    # device = torch.device(device)

    model.eval()

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    for i in range(ntokens):
        with torch.no_grad():
            if torch.cuda.is_available():
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits = model(x)
            else:
                logits = model(x)  # B, T, vocab_size

            if type(logits) != torch.Tensor:
                # in case an HF model was given
                logits = logits.logits

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

    model.train()

    if return_new:
        x = x[:, T:]

    if return_format == "text":
        ret_text = []
        for i in range(x.shape[0]):
            dec_tokens = list(x[i, :])
            ret_text.append(enc.decode(dec_tokens))
        return ret_text
    elif return_format == "tokens":
        return x
    else:
        return None


@dataclass
class GPTConfig:
    vocab_size: int = 50257
    n_embed: int = 768
    max_seq_length: int = 1024
    n_layer: int = 12
    n_heads: int = 12
