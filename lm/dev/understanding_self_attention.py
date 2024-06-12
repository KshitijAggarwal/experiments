# Coding up self attention
# ref: https://www.youtube.com/watch?v=kCc8FmEb1nY

# %%

import torch

torch.manual_seed(42)
B, T, C = 4, 8, 2  # Batch, time, channels

x = torch.randn(B, T, C)

# %%
# We want to collect information from
# previous tokens till current token
# and not include future tokens as we haven't seen them yet
# so x[b, t] = mean_{i<=t} x[b, i]
xbow = torch.zeros((B, T, C))
for b in range(B):
    for t in range(T):
        xprev = x[b, : t + 1]  # (t, C)
        xbow[b, t] = torch.mean(xprev, 0)

# %%
# using matrix multiplication

torch.tril(torch.ones(3, 3))

# tensor([[1., 0., 0.],
#         [1., 1., 0.],
#         [1., 1., 1.]])

torch.manual_seed(42)
a = torch.tril(torch.ones(3, 3))
b = torch.randint(0, 10, (3, 2)).float()
c = a @ b

print(a)
print(b)
print(c)

# %%
# normalizing elements of a

torch.manual_seed(42)
a = torch.tril(torch.ones(3, 3))
a = a / torch.sum(a, dim=1, keepdim=True)

b = torch.randint(0, 10, (3, 2)).float()
c = a @ b

print(a)
print(b)
print(c)

# %%

wei = torch.tril(torch.ones(T, T))
wei = wei / torch.sum(wei, dim=1, keepdim=True)

# x shape -> B, T, C
# wei shape -> T, T
# wei @ x -> pytorch will add a batch dim to wei
# so, (B, T, T) @ (B, T, C)
# @ is a batched matrix multiplication
# so B (T, T) x (T, C) -> (B, T, C)

xbow2 = wei @ x

# This is basically a weighted sum.
# Torch.tril is makign sure that tokens only see tokens before it.
# wei / torch.sum(..) is giving weights.
# So wei @ x would take the weighted sum -> average across time per row

# %%
# using softmax
import torch.nn.functional as F

tril = torch.tril(torch.ones((T, T)))

wei = torch.zeros((T, T))  # here we set them as zero
# but they can be affinities between tokens!

wei = wei.masked_fill(tril == 0, float("-inf"))
wei = F.softmax(wei, dim=-1)  # remember: softmax exponentiates and normalizes
xbow3 = wei @ x

# %%
import torch
import torch.nn.functional as F

torch.manual_seed(42)
B, T, C = 4, 8, 32  # Batch, time, channels
x = torch.randn(B, T, C)

tril = torch.tril(torch.ones((T, T)))  # only look at past tokens
wei = torch.zeros(
    (T, T)
)  # tokens are independent/uniform. No affinities between tokens
wei = wei.masked_fill(tril == 0, float("-inf"))  # setting future to -inf
wei = F.softmax(wei, dim=-1)  # softmax to get weights for averaging along time
out = wei @ x  # actual weighted sum or average along time

# %%
# Self-attention
# every single token will emit 2 vectors:
# query and key
# query: what am I looking for
# key: what do I contain
# dot product between key and queries -> affinities between tokens

import torch
import torch.nn.functional as F
import torch.nn as nn

torch.manual_seed(42)
B, T, C = 4, 8, 32  # Batch, time, channels
x = torch.randn(B, T, C)


# single head of self attention
head_size = 16
key = nn.Linear(C, head_size, bias=False)
query = nn.Linear(C, head_size, bias=False)
value = nn.Linear(C, head_size, bias=False)

k = key(x)  # (B, T, head_size)
q = query(x)  # (B, T, head_size)

# no communication between tokens happened yet
wei = q @ k.transpose(1, 2) * C ** (-0.5)
# (B, T, T) # now tokens communicate with each other to get affinities
# C ** (-0.5) is for scaling so that variance of wei is close to 1.
# otherwise after softmax it can get too peaky and not diffuse, esp at initialization

# wei = torch.zeros(
#     (T, T)
# )  # tokens are independent/uniform. No affinities between tokens

tril = torch.tril(torch.ones((T, T)))  # only look at past tokens
wei = wei.masked_fill(tril == 0, float("-inf"))  # setting future to -inf
wei = F.softmax(wei, dim=-1)  # softmax to get weights for averaging along time

# >> wei
# tensor([[1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
#         [0.1905, 0.8095, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
#         [0.3742, 0.0568, 0.5690, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
#         [0.1288, 0.3380, 0.1376, 0.3956, 0.0000, 0.0000, 0.0000, 0.0000],
#         [0.4311, 0.0841, 0.0582, 0.3049, 0.1217, 0.0000, 0.0000, 0.0000],
#         [0.0537, 0.3205, 0.0694, 0.2404, 0.2568, 0.0592, 0.0000, 0.0000],
#         [0.3396, 0.0149, 0.5165, 0.0180, 0.0658, 0.0080, 0.0373, 0.0000],
#         [0.0165, 0.0375, 0.0144, 0.1120, 0.0332, 0.4069, 0.3136, 0.0660]],
#        grad_fn=<SelectBackward0>)

# Now we can see that the weights aren't uniform.
# Token at a position can have a higher/lower affinity for a token at a different position
# and that will then impact the weighted sum in the next step

v = value(x)  # (B, T, head_size)
out = wei @ v  # actual weighted sum or average along time
out.shape  # B, T, head_size

# %%
