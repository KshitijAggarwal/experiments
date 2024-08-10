import torch 
import pylab as plt 

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
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # Return the mean KL divergence over the batch
    return kl_loss

def visualize_reconstruction(model, test_loader, device, num_images=10, title=''):
    """
    Visualize the reconstruction of a model on a given test dataset.

    Args:
    - model: The model to visualize the reconstruction of.
    - test_loader: The test dataset loader.
    - device: The device to run the model on.
    - num_images: The number of images to visualize. Defaults to 10.
    - title: The title of the visualization plot. Defaults to an empty string.

    Returns:
    - None
    """
    model.eval()
    with torch.no_grad():
        data, _ = next(iter(test_loader))
        data = data[:num_images].view(num_images, -1).to(device)
        recon = model(data, return_intermediate=False)

        data = data.cpu().view(num_images, 1, 28, 28)
        recon = recon.cpu().view(num_images, 1, 28, 28)

        fig, axes = plt.subplots(2, num_images, figsize=(20, 4))
        for i in range(num_images):
            axes[0, i].imshow(data[i].squeeze(), cmap='gray')
            axes[1, i].imshow(recon[i].squeeze(), cmap='gray')
        plt.suptitle(title)
        plt.show()
