from torch import nn 
import torch 

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
            
    def forward(self, x, return_intermediate=True):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        if return_intermediate:
            return encoded, decoded
        else:
            return decoded
    
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

    def forward(self, x, return_intermediate=True):
        encoded = self.encoder(x)

        mus = self.mu(encoded)
        log_var = self.log_var(encoded)

        z = self.reparameterize(mus, log_var)
        
        decoded = self.decoder(z)

        if return_intermediate:
            return encoded, decoded, mus, log_var
        else:
            return decoded

    def generate(self, z=None, num_samples=2, device='cpu'):
        if z is None:
            z = torch.randn(num_samples, self.config.latent_dim).to(device)
        with torch.no_grad():
            decoded = self.decoder(z)
        return decoded


if __name__ == "__main__":
    from dataclasses import dataclass
    import torch 

    @dataclass
    class VAEConfig:
        image_size: tuple = (28, 28) # H, W
        n_channels: int = 1
        latent_dim: int = 2
        hidden_dim: int = 32
        encoder_layers: int = 2
        decoder_layers: int = 2

    config = VAEConfig()

    B = 2
    C = config.n_channels
    H = config.image_size[0]
    W = config.image_size[1]

    image = torch.rand(((B, C, H, W))) # (B, C, H, W)
    flattened_image = image.reshape((B, C*H*W))

    model = AutoEncoder(config)
    enc, dec = model(flattened_image)
    print(image.shape, enc.shape, dec.shape)

    model = VAE(config)
    enc, dec, mus, log_var, z = model(flattened_image)
    print(image.shape, enc.shape, dec.shape, mus.shape, log_var.shape)
    

