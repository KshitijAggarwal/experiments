# %%
# "val": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl",
# https://github.com/karpathy/build-nanogpt/blob/master/hellaswag.py

import json
import tiktoken
import torch
from lm.utils import gen_text, GPTConfig
import tiktoken
from lm.gpt2 import GPT2
import tqdm
import os

basepath = "/Users/kshitijaggarwal/Documents/Projects/experiments/data/outputs/"

filename = basepath + "model_GPT_19073_interp_29M.pt"
# filename = basepath + "model_GPT_04000_interp_17M.pt"


def remove_orig_mod(state_dict):
    return {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}


if not os.path.isfile(filename):
    raise FileNotFoundError(f"No checkpoint found at {filename}")

checkpoint = torch.load(filename, map_location=torch.device("cpu"))
config = GPTConfig(**checkpoint["model_config"])
model = GPT2(config)

if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

model.load_state_dict(remove_orig_mod(checkpoint["model_state_dict"]))

from transformers import GPT2LMHeadModel

model = GPT2LMHeadModel.from_pretrained("gpt2")

model.to(device)

# %%

file = "/Users/kshitijaggarwal/Documents/Projects/experiments/data/hellaswag_val.jsonl"

with open(file, "r") as f:
    data = f.readlines()

data = [eval(d) for d in data]

enc = tiktoken.get_encoding("gpt2")

# %%

import torch.nn.functional as F

show_correct_examples = False
correct_count = 0
for d in tqdm.tqdm(data):
    context = d["ctx"]
    endings = d["endings"]
    label = d["label"]

    context_tokens = enc.encode(context)

    ending_losses = []
    encoded_endings = [enc.encode(" " + ending) for ending in endings]
    max_len = max([len(l) for l in encoded_endings])

    input_tokens = torch.zeros([4, max_len + len(context_tokens)], dtype=torch.long)
    mask = torch.zeros([4, max_len + len(context_tokens)], dtype=torch.long)

    for i, ending in enumerate(endings):
        input_tokens[i, : len(context_tokens)] = torch.tensor(context_tokens)
        l = len(encoded_endings[i])
        input_tokens[i, len(context_tokens) : len(context_tokens) + l] = torch.tensor(
            encoded_endings[i]
        )
        mask[i, : len(context_tokens) + l] = (
            1  # 1 where context and ending tokens are present
        )

    input_tokens = input_tokens.to(device)
    with torch.no_grad():
        logits = model(input_tokens)

    if type(logits) != torch.Tensor:
        # in case an HF model was given
        logits = logits.logits

    logits = logits[
        :, :-1, :
    ].contiguous()  # last token is for prediction beyond context. Not needed here.
    mask = mask[
        :, :-1
    ].contiguous()  # last token is for prediction beyond context. Not needed here.

    B, T, C = logits.shape
    logits = logits.view(B * T, C)
    targets = (
        input_tokens[:, 1:].contiguous().view(B * T)
    )  # Target is from second token till the end of context.

    ending_loss = (
        F.cross_entropy(logits, targets, reduction="none").view(B, T).detach().cpu()
    )

    avg_loss = (ending_loss * mask).sum(-1) / mask.sum(
        -1
    )  # average loss for each ending based on where mask is 1

    if int(avg_loss.argmin()) == label:
        correct_count += 1

        # print a random case with correct prediction
        if torch.rand(1)[0] < 0.1 and show_correct_examples:
            print(f"Context: {context}")
            for i, e in enumerate(endings):
                print(f"Ending {i}: {e}")
            print(f"label: {label}")
            print("----")

# %%

print(f"Accuracy of {filename.split('/')[-1]}: {correct_count / len(data):.2f}")

# %%
# Accuracy of model_GPT_04000_interp_17M.pt: 0.24
# Accuracy of model_GPT_19073_interp_29M.pt: 0.24
# Accuracy of HF GPT-2 (124M): 0.25

# %%
