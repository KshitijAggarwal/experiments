# %%

import torch
import os
from lm.utils import gen_text, GPTConfig

def remove_orig_mod(state_dict):
    return {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

filename = '/workspace/log/model_GPT_19073_interp_29M.pt'
# filename = '/workspace/log/model_GPT_04000_interp_17M.pt'


if not os.path.isfile(filename):
    raise FileNotFoundError(f"No checkpoint found at {filename}")

checkpoint = torch.load(filename)

config = GPTConfig(**checkpoint["model_config"])
from lm.gpt2 import GPT2
model = GPT2(config)

# %%

model.load_state_dict(remove_orig_mod(checkpoint["model_state_dict"]))
model.to('cuda')
max_length = 32
gen_text(model, 5, max_length, "cuda", 
         prefix="Hi, I am a helpful AI assistant. ")

# %%
