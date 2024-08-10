# %% 

from torch import nn 
import torch 
from dataclasses import dataclass
from torch import optim 

# %% 

class Linear(nn.Module):
    def __init__(self, input_dim, hidden_dim, act=nn.ReLU()):
        super().__init__()

        self.linear = nn.Linear(input_dim, hidden_dim, bias=True)
        self.act = act
    
    def forward(self, x):
        return self.act(self.linear(x))


class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim, n_layers):
        super().__init__()
        
        self.n_layers = n_layers
        n_hidden = self.n_layers - 2 

        self.first_layer = Linear(input_dim, hidden_dim)
        self.last_layer = Linear(hidden_dim, latent_dim)
        self.hidden_layers = nn.ModuleList([
            Linear(hidden_dim, hidden_dim) for _ in range(n_hidden)
        ])

    def forward(self, x):
        x = self.first_layer(x)
        for layer in self.hidden_layers:
            x = layer(x)
        x = self.last_layer(x)
        return x

class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim, hidden_dim, n_layers):
        super().__init__()
        
        self.n_layers = n_layers
        n_hidden = self.n_layers - 2 

        self.first_layer = Linear(latent_dim, hidden_dim)
        self.last_layer = Linear(hidden_dim, output_dim, act=nn.Sigmoid())
        self.hidden_layers = nn.ModuleList([
            Linear(hidden_dim, hidden_dim) for _ in range(n_hidden)
        ])

    def forward(self, x):
        x = self.first_layer(x)
        for layer in self.hidden_layers:
            x = layer(x)
        x = self.last_layer(x)
        return x

class AutoEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        C = config.n_channels
        H = config.image_size[0]
        W = config.image_size[1]
        latent_dim = config.latent_dim

        self.encoder = Encoder(input_dim=C*H*W, 
                               latent_dim=latent_dim,
                               hidden_dim=config.hidden_dim,
                               n_layers=config.encoder_layers)
        
        self.decoder = Decoder(latent_dim=latent_dim,
                               hidden_dim=config.hidden_dim,
                               n_layers=config.decoder_layers,
                               output_dim=C*H*W)
            
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded 

class VAE(AutoEncoder):
    def __init__(self, config):
        super().__init__(config=config)
        self.config = config
        self.mu = nn.Linear(config.latent_dim, config.latent_dim)
        self.log_var = nn.Linear(config.latent_dim, config.latent_dim)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        # randn_like: Returns a tensor with the same size as input that is 
        # filled with random numbers from a normal distribution with mean 
        # 0 and variance 1. 
        eps = torch.randn_like(std) 
        return mu + eps * std

    def forward(self, x):
        encoded = self.encoder(x)

        mus = self.mu(encoded)
        log_var = self.log_var(encoded)

        z = self.reparameterize(mus, log_var)
        
        decoded = self.decoder(z)
        return encoded, decoded, mus, log_var

    def generate(self, num_samples, device, z):
        with torch.no_grad():
            if not z:
                z = torch.randn(num_samples, self.config.latent_dim).to(device)
            decoded = self.decoder(z)
        return decoded

# %% 

@dataclass
class VAEConfig:
    image_size: tuple = (28, 28) # H, W
    n_channels: int = 1
    latent_dim: int = 8
    hidden_dim: int = 32
    encoder_layers: int = 2
    decoder_layers: int = 2

config = VAEConfig()

B = 32
C = config.n_channels
H = config.image_size[0]
W = config.image_size[1]

image = torch.rand(((B, C, H, W))) # (B, C, H, W)
flattened_image = image.reshape((B, C*H*W))

# %% 

# model = AutoEncoder(config)
# enc, dec = model(flattened_image)

model = VAE(config)
enc, dec, mus, log_var = model(flattened_image)

# %% 

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch 

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(f"Using device: {device}")

# %% 

# MNIST data
transform = transforms.Compose([
    transforms.ToTensor(),
])

data_dir = '/Users/kshitijaggarwal/Documents/Projects/experiments/data/'

train_dataset = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)

batch_size = 32

train_dataloader = DataLoader(batch_size=batch_size, dataset=train_dataset, shuffle=True)
test_dataloader = DataLoader(batch_size=batch_size, dataset=test_dataset, shuffle=False)

# %% 

b = next(iter(train_dataloader))
print(b[0].shape, b[1].shape)

# %% 

nepochs = 100

optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

def kl_divergence(mu, logvar):
    """
    Compute the KL divergence between a Gaussian distribution with mean `mu` and variance `exp(logvar)`
    and a standard normal distribution N(0, I).

    Args:
    - mu (torch.Tensor): The mean of the learned Gaussian distribution. Shape: (batch_size, latent_dim)
    - logvar (torch.Tensor): The log variance (log of the diagonal of the covariance matrix) of the learned Gaussian distribution. Shape: (batch_size, latent_dim)

    Returns:
    - torch.Tensor: The KL divergence loss, summed over the latent dimensions and averaged over the batch.
    """
    # Compute KL divergence
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    
    # Return the mean KL divergence over the batch
    return kl_loss.mean()

beta = 1

model.to(device)
# training 
for epoch in range(nepochs):
    model.train()
    train_loss = 0
    for images, labels in train_dataloader:
        images = images.reshape((images.shape[0], -1)).to(device)

        optimizer.zero_grad()
        # enc, dec = model(images)
        enc, dec, mus, log_var = model(images)

        recon_loss = loss_fn(images, dec)
        kl_loss = kl_divergence(mus, logvar=log_var)

        loss = recon_loss + beta * kl_loss
        loss.backward()

        optimizer.step()
        train_loss += loss.item()

    avg_loss = train_loss / len(train_dataloader)
    print(f'Epoch {epoch+1}, Average loss: {avg_loss:.4f}')


# %% 

import pylab as plt

def visualize_reconstruction(model, test_loader, num_images=10, title=''):
    model.eval()
    with torch.no_grad():
        data, _ = next(iter(test_loader))
        data = data[:num_images].view(num_images, -1).to(device)
        enc, recon = model(data)

        data = data.cpu().view(num_images, 1, 28, 28)
        recon = recon.cpu().view(num_images, 1, 28, 28)

        fig, axes = plt.subplots(2, num_images, figsize=(20, 4))
        for i in range(num_images):
            axes[0, i].imshow(data[i].squeeze(), cmap='gray')
            axes[1, i].imshow(recon[i].squeeze(), cmap='gray')
        plt.suptitle(title)
        plt.show()


# %% 

visualize_reconstruction(model, test_dataloader, num_images=10, title='AutoEncoder Reconstruction')

# %% 

model.eval()
with torch.no_grad():
    test_embeddings = []
    labels = []
    for images, label in test_dataloader:
        images = images.reshape((images.shape[0], -1)).to(device)
        labels.append(label.numpy())
        enc, dec = model(images)
        test_embeddings.append(enc.cpu().numpy())

# %% 

import numpy as np 

test_embeddings = np.concatenate(test_embeddings)
labels = np.concatenate(labels)

# %% 

plt.figure(figsize=(8, 6))

# Create a scatter plot for each digit
for digit in range(10):
    mask = labels == digit
    plt.scatter(test_embeddings[mask, 0], test_embeddings[mask, 1], 
                label=str(digit), alpha=0.6, s=15)

plt.legend()
plt.title("2D Embeddings of MNIST Digits from AutoEncoder")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.tight_layout()
plt.show()

# %% 

num_images = 5
dummy_encodings = torch.rand((num_images, config.latent_dim)) * 20
with torch.no_grad():
    decoded = model.decoder(dummy_encodings.to(device)).cpu().numpy()

dec_image = decoded.reshape((num_images, config.image_size[0], config.image_size[1]))

fig, axes = plt.subplots(1, num_images, figsize=(20, 4))
plt.suptitle('Sampling latent space from AutoEncoder (latent dim=8)')
for i in range(num_images):
    axes[i].imshow(dec_image[i], cmap='gray')
plt.tight_layout()

# %% 



# %%

