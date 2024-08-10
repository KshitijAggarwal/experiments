# %% 

import torch

# Set up requires_grad for our variables
mu = torch.tensor([0.0], requires_grad=True)
log_var = torch.tensor([0.0], requires_grad=True)

# Direct sampling
z_direct = torch.normal(mu, torch.exp(0.5 * log_var))
loss_direct = z_direct.sum()
loss_direct.backward()

print("Direct sampling gradients:")
print("mu.grad:", mu.grad)
print("log_var.grad:", log_var.grad)

# Reset gradients
mu.grad.zero_()
log_var.grad.zero_()

# Reparameterization trick
epsilon = torch.randn_like(mu)
z_reparam = mu + torch.exp(0.5 * log_var) * epsilon
loss_reparam = z_reparam.sum()
loss_reparam.backward()

print("\nReparameterization trick gradients:")
print("mu.grad:", mu.grad)
print("log_var.grad:", log_var.grad)

# %% 

# When you use torch.normal(), PyTorch doesn't know how to compute gradients 
# through this random sampling operation. During backpropagation, 
# the gradients essentially stop at this point. The network can't learn 
# how changes in mu and log_var affect the final loss.

# Reparameterization Trick:
# This method expresses the random sampling as a deterministic function of mu,
# log_var, and a separate random variable epsilon. Now, PyTorch can compute 
# gradients with respect to mu and log_var.
