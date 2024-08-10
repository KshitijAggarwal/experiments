from torch import nn 

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
