# %%

from transformers import pipeline, set_seed

generator = pipeline("text-generation", model="gpt2")
set_seed(42)
generator("Hello, I'm a language model,", max_length=30, num_return_sequences=5)

# %%

from transformers import GPT2Tokenizer, GPT2Model

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2Model.from_pretrained("gpt2")
text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors="pt")
output = model(**encoded_input)

# [{'generated_text': "Hello, I'm a language model, I'm writing a new language for you. But first, I'd like to tell you about the language itself"},
#  {'generated_text': "Hello, I'm a language model, and I'm trying to be as expressive as possible. In order to be expressive, it is necessary to know"},
#  {'generated_text': "Hello, I'm a language model, so I don't get much of a license anymore, but I'm probably more familiar with other languages on that"},
#  {'generated_text': "Hello, I'm a language model, a functional model... It's not me, it's me!\n\nI won't bore you with how"},
#  {'generated_text': "Hello, I'm a language model, not an object model.\n\nIn a nutshell, I need to give language model a set of properties that"}]

# %%

from transformers import GPT2LMHeadModel

model_hf = GPT2LMHeadModel.from_pretrained("gpt2")

# %%

sd_hf = model_hf.state_dict()

for k, v in sd_hf.items():
    print(k, v.shape)

# transformer.wte.weight torch.Size([50257, 768]) -> vocab size for GPT-2 was 50257
# transformer.wpe.weight torch.Size([1024, 768]) -> sequence length for GPT-2 was 1024
# transformer.h.0.ln_1.weight torch.Size([768]) -> GPT-2 embeding dim was 768
# transformer.h.0.ln_1.bias torch.Size([768])
# ...
# lm_head.weight torch.Size([50257, 768])

# %%

sd_hf["transformer.wte.weight"].shape

# %%

import pylab as plt

plt.imshow(sd_hf["transformer.wpe.weight"], cmap="gray")
# has structure.
# Each row is a position.
# Position encoding in GPT-2 were learnt.

# %%

plt.plot(sd_hf["transformer.wpe.weight"][:, 10])
plt.plot(sd_hf["transformer.wpe.weight"][:, 100])
plt.plot(sd_hf["transformer.wpe.weight"][:, 200])
plt.xlabel("Token Position")

# %%

# Structure in weights
print(sd_hf["transformer.h.1.attn.c_attn.weight"].shape)
plt.imshow(sd_hf["transformer.h.1.attn.c_attn.weight"][:500, :500], cmap="gray")

# %%

import torch

wte = sd_hf["transformer.wte.weight"]
lm_head = sd_hf["lm_head.weight"]

print(torch.equal(wte, lm_head))

# weights are tied.
# token embedding is tied to lm_head

# %%
