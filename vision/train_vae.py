# %% 

from vision.vae import VAE
from torch import optim 
from torch import nn 
from dataclasses import dataclass
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch 
from vision.utils import kl_divergence, visualize_reconstruction
from misc.utils import get_device 

@dataclass
class VAEConfig:
    image_size: tuple = (28, 28) # H, W
    n_channels: int = 1
    latent_dim: int = 2
    hidden_dim: int = 256
    encoder_layers: int = 2
    decoder_layers: int = 2

config = VAEConfig()
model = VAE(config)

device = get_device()

# MNIST data
transform = transforms.Compose([
    transforms.ToTensor(),
])

data_dir = '/Users/kshitijaggarwal/Documents/Projects/experiments/data/'

train_dataset = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)

batch_size = 64

train_dataloader = DataLoader(batch_size=batch_size, dataset=train_dataset, shuffle=True)
test_dataloader = DataLoader(batch_size=batch_size, dataset=test_dataset, shuffle=False)

nepochs = 100

optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss(reduction='sum')

beta = 1

model.to(device)
# training 
for epoch in range(nepochs):
    model.train()
    train_loss = 0
    recon_losses = 0
    kl_losses = 0
    for images, labels in train_dataloader:
        images = images.reshape((images.shape[0], -1)).to(device)

        optimizer.zero_grad()
        # enc, dec = model(images)
        _, dec, mus, log_var = model(images)

        recon_loss = loss_fn(images, dec)
        kl_loss = kl_divergence(mus, log_var)

        loss = recon_loss + beta * kl_loss
        loss.backward()

        optimizer.step()

        train_loss += loss.item()
        recon_losses += recon_loss.item()
        kl_losses += kl_loss.item()

    avg_loss = train_loss / len(train_dataloader)
    avg_recon_loss = recon_losses / len(train_dataloader)
    avg_kl_loss = kl_losses / len(train_dataloader)
    print(f'Epoch {epoch+1}, Average loss: {avg_loss:.5f}, Reconstruction loss: {avg_recon_loss:.5f}, KL loss: {avg_kl_loss:.5f}')

# %% 

visualize_reconstruction(model, test_dataloader, 
                         device, num_images=10, 
                         title='VAE Reconstruction')

# %% 
# Plot 2D embeddings of MNIST digits

import numpy as np 
import pylab as plt 

model.eval()
with torch.no_grad():
    test_embeddings = []
    labels = []
    for images, label in test_dataloader:
        images = images.reshape((images.shape[0], -1)).to(device)
        labels.append(label.numpy())
        enc, dec, _, _ = model(images)
        test_embeddings.append(enc.cpu().numpy())

test_embeddings = np.concatenate(test_embeddings)
labels = np.concatenate(labels)

plt.figure(figsize=(8, 6))

# Create a scatter plot for each digit
for digit in range(10):
    mask = labels == digit
    plt.scatter(test_embeddings[mask, 0], test_embeddings[mask, 1], 
                label=str(digit), alpha=0.6, s=15)

plt.legend()
plt.title("2D Embeddings of MNIST Digits from VAE")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.tight_layout()
plt.show()

# %% 
## Sampling latent space from VAE
num_images = 10
dummy_encodings = torch.rand((num_images, config.latent_dim)) * 10
with torch.no_grad():
    decoded = model.decoder(dummy_encodings.to(device)).cpu()
# decoded = model.generate(z=dummy_encodings, device=device)
dec_image = decoded.reshape((num_images, config.image_size[0], config.image_size[1]))

fig, axes = plt.subplots(1, num_images, figsize=(20, 4))
plt.suptitle('Sampling latent space from VAE (latent dim=2)')
for i in range(num_images):
    axes[i].imshow(dec_image[i], cmap='gray')
plt.tight_layout()

# %% 
## Smoothly varying one dim of latent space

n_samples = 10  
constant_value = 5
row_constant = torch.full((n_samples,), constant_value)
row_variable = torch.linspace(0, 20, n_samples)
# Stack the two rows to form a 2 x n_samples tensor
dummy_encodings = torch.stack([row_constant, row_variable]).T

with torch.no_grad():
    decoded = model.decoder(dummy_encodings.to(device)).cpu()

dec_image = decoded.reshape((num_images, config.image_size[0], config.image_size[1]))

fig, axes = plt.subplots(1, num_images, figsize=(20, 4))
plt.suptitle('Smoothly varying latent space for VAE (latent dim=2)')
for i in range(num_images):
    axes[i].imshow(dec_image[i], cmap='gray')
plt.tight_layout()

# %% 