import math
import pylab as plt
import numpy as np


def get_lr(step, warmup_steps, max_steps, max_lr, min_lr):
    # warmup + cosine decay LR schedule
    # 1) Linear warmup
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps

    # 2) if step > max_steps, return min_lr
    if step > max_steps:
        return min_lr

    # 3) cosine decay in between the above two
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= progress <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + coeff * (max_lr - min_lr)


nsteps = 1000
steps = np.linspace(0, nsteps, nsteps)
max_lr = 6e-4
min_lr = max_lr * 0.1

lrs = []
for step in steps:
    lr = get_lr(step, 0.1 * nsteps, nsteps, max_lr, min_lr)
    lrs.append(lr)

plt.plot(steps, lrs)
plt.xlabel("step")
plt.ylabel("lr")
plt.grid()
plt.show()
