# %%
import torch
import os
from lm.utils import gen_text, GPTConfig

filename = (
    "/Users/kshitijaggarwal/Documents/Projects/experiments/lm/log/model_GPT2_00009.pt"
)

if not os.path.isfile(filename):
    raise FileNotFoundError(f"No checkpoint found at {filename}")

checkpoint = torch.load(filename)


def load_checkpoint(checkpoint, model, optimizer, **kwargs):

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # Load scheduler state if it exists and a scheduler is provided
    if checkpoint["scheduler_state_dict"] and kwargs.get("scheduler"):
        kwargs["scheduler"].load_state_dict(checkpoint["scheduler_state_dict"])

    # Set random states if needed
    if kwargs.get("set_rng_state", False):
        torch.set_rng_state(checkpoint["torch_rng_state"])

    print(f"Checkpoint loaded from {filename}")
    return model, optimizer


# %%

config = GPTConfig(**checkpoint["model_config"])
if checkpoint["model_name"] == "GPT2":
    from lm.gpt2 import GPT2

    model = GPT2(config)
else:
    from lm.bitnet import BitNet

    model = BitNet(config)

optimizer = model.configure_optimizers(
    weight_decay=0.01, learning_rate=6e-4, device_type="cpu"
)

# %%

# optimizer = torch.optim.AdamW(model.parameters())
model, optimizer = load_checkpoint(
    checkpoint,
    model,
    optimizer,
    set_rng_state=True,
)

# %%

max_length = 32
gen_text(model, 5, max_length, "cpu")

# %%
