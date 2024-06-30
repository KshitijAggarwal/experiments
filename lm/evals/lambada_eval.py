# %%
# https://huggingface.co/datasets/cimec/lambada/blob/main/plain_text/test-00000-of-00001.parquet

import pandas as pd
from lm.utils import gen_text, GPTConfig
import tiktoken
import os
import torch
from lm.gpt2 import GPT2
import tqdm

df = pd.read_parquet(
    "/Users/kshitijaggarwal/Documents/Projects/experiments/data/test-00000-of-00001_lambada.parquet"
)

basepath = "/Users/kshitijaggarwal/Documents/Projects/experiments/data/outputs/"

# filename = basepath + "model_GPT_19073_interp_29M.pt"
filename = basepath + "model_GPT_04000_interp_17M.pt"

# %%


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

# from transformers import GPT2LMHeadModel
# model_hf = GPT2LMHeadModel.from_pretrained("gpt2")

model.to(device)

enc = tiktoken.get_encoding("gpt2")

# %%
# Generate 5 predictions of length 2 tokens for each context
# Search for the target word in the predictions

# device = "mps"
# count = 0
# for i, row in tqdm.tqdm(df.iterrows(), total=len(df)):
#     # print(row["text"])

#     context = " ".join(row["text"].split(" ")[:-1])
#     target = row["text"].split(" ")[-1]
#     # print("Target: ", target)
#     text = gen_text(
#         model,
#         num_return_sequences=5,
#         ntokens=2,
#         device=device,
#         prefix=context,
#         return_format="text",
#         return_new=True,
#     )

#     for res in text:
#         pred = res.split(" ")
#         if target in pred:
#             # print("Context: ", context)
#             # print("Target: ", target)
#             # print("Predicted: ", pred)
#             count += 1
#             break

# print(f"Accuracy of {filename.split('/')[-1]}: {count / len(df)}")

# %%
# Generate 5 predictions of length(target tokens) for each context
# Convert the target into tokens and compare the tokens with the predicted tokens
# Would be faster than comparing text, as we don't need to decode before comparison

device = "mps"
count = 0
for i, row in tqdm.tqdm(df.iterrows(), total=len(df)):
    context = " ".join(row["text"].split(" ")[:-1])
    target = row["text"].split(" ")[-1]
    target_tokens = torch.tensor(enc.encode(" " + target))

    pred_tokens = gen_text(
        model,
        num_return_sequences=5,
        ntokens=len(target_tokens),
        device=device,
        prefix=context,
        return_format="tokens",
        return_new=True,
    )

    if pred_tokens.shape[1] != len(target_tokens):
        break

    for res in pred_tokens.detach().cpu():
        if (target_tokens == res).all():
            # print("Context: ", context)
            # print("Target: ", target)
            # print("Predicted: ", enc.decode(list(res)))
            count += 1
            break

print(f"Accuracy of {filename.split('/')[-1]}: {count / len(df)}")


# %%
# Results
# Accuracy of model_GPT_19073_interp_29M.pt: 0.020958664855424025
# Accuracy of model_GPT_04000_interp_17M.pt: 0.08713370851930914
# Accuracy of HF GPT-2: 0.0 # Not sure why..
