import torch


def straight_through(x):
    return x + (torch.round(x) - x).detach()


# Usage
x = torch.tensor([0.2, 0.7, 1.2, 1.9], requires_grad=True)
y = straight_through(x)

print("x:", x)
print("y:", y)

# Backpropagation
y.sum().backward()

print("x.grad:", x.grad)
