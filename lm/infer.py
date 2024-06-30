# %%

import torch
import os
from lm.utils import gen_text, GPTConfig


def remove_orig_mod(state_dict):
    return {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}


basepath = "/Users/kshitijaggarwal/Documents/Projects/experiments/data/outputs/"

# filename = basepath + "model_GPT_19073_interp_29M.pt"
filename = basepath + "model_GPT_04000_interp_17M.pt"


if not os.path.isfile(filename):
    raise FileNotFoundError(f"No checkpoint found at {filename}")

checkpoint = torch.load(filename, map_location=torch.device("cpu"))

config = GPTConfig(**checkpoint["model_config"])
from lm.gpt2 import GPT2

model = GPT2(config)

# %%

if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

model.load_state_dict(remove_orig_mod(checkpoint["model_state_dict"]))

# from transformers import GPT2LMHeadModel
# model = GPT2LMHeadModel.from_pretrained("gpt2")

model.to(device)

# %%

max_length = 64
gen_text(
    model,
    num_return_sequences=5,
    max_length=max_length,
    device=device,
    prefix="Plants are good. ",
)

# %%

ntokens = 20
gen_text(
    model,
    num_return_sequences=5,
    ntokens=ntokens,
    device=device,
    prefix="President of US is going to be ",
)

# %%
