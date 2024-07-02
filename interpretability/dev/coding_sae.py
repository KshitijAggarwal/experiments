# %%

from torch import nn
import torch
import torch.nn.functional as F
import os
from lm.gpt2 import GPT2
from lm.utils import GPTConfig, ddp_setup
from lm.data import DataLoaderLite
import torch.optim as optim

# %%
ddp, device, ddp_rank, ddp_local_rank, ddp_world_size, master_process = ddp_setup()

device_type = "mps"

basepath = "/Users/kshitijaggarwal/Documents/Projects/experiments/data/outputs/"
# filename = basepath + "model_GPT_19073_interp_29M.pt"
filename = basepath + "model_GPT_04000_interp_17M.pt"

train_files = [
    "/Users/kshitijaggarwal/Documents/Projects/experiments/data/sample-1MT.npy"
]


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
model.to(device)

# %%


def get_activations(model, idx):
    B, T = idx.shape
    pe = torch.arange(0, T, dtype=torch.long, device=idx.device)  # shape (T)
    wpe = model.transformer.wpe(pe)  # position embeddings. Shape: T, n_embed
    wte = model.transformer.wte(idx)  # word token embeddings. Shape: B, T, n_embed

    x = wte + wpe  # note the hidden broadcasting happening here.

    for block in model.transformer.h:
        x = block(x)
    # x = self.transformer.ln_f(x)  # B, T, n_embed

    # logits = self.lm_head(x)  # B, T, vocab_size
    return x


# %%


class SparseAutoencoder(nn.Module):
    def __init__(self, n_embed, sae_dim):
        super(SparseAutoencoder, self).__init__()
        self.encoder = nn.Linear(n_embed, sae_dim, bias=True)
        self.decoder = nn.Linear(sae_dim, n_embed, bias=False)

        # Initialize the decoder matrix with orthogonal initialization
        nn.init.orthogonal_(self.decoder.weight)

    def forward(self, x):
        B, T, n_embed = x.shape
        c = torch.relu(self.encoder(x)).view(B * T, -1)

        # Normalize the decoder columns to have unit norm
        D = self.decoder.weight / torch.norm(self.decoder.weight, dim=0, keepdim=True)

        # Decoder
        x_reconstructed = torch.matmul(c, D.t()).view(B, T, n_embed)

        return x_reconstructed, c


# %%

sae_dim = config.n_embed * 4
learning_rate = 0.01
l1_penalty = 0.1
max_steps = 1000

sae = SparseAutoencoder(config.n_embed, sae_dim).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(sae.parameters(), lr=learning_rate)

B = 64
T = 128

train_loader = DataLoaderLite(
    B=B,
    T=T,
    process_rank=ddp_local_rank,
    num_processes=ddp_world_size,
    tokenfiles=train_files,
)

dataloader = train_loader

# %%

model.eval()
sae.train()

for step in range(max_steps):
    x, y = dataloader.next_batch()
    x, y = x.to(device), y.to(device)

    with torch.no_grad():
        if torch.cuda.is_available():
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                act = get_activations(model, x)
        else:
            act = get_activations(model, x)

    # for step in range(1000):
    optimizer.zero_grad()

    act_recon, c = sae(act)
    rec_loss = criterion(act_recon, act)
    l1_loss = l1_penalty * torch.mean(torch.abs(c))
    loss = rec_loss + l1_loss

    loss.backward()
    optimizer.step()

    if step % 50 == 0 and step > 0:
        print(f"Step {step}, Loss: {loss.item():.4f}")

# %%
