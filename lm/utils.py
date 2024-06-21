import math


def get_lr(step, warmup_steps, max_steps, max_lr, min_lr):
    """
    Calculate the learning rate for a given step in a warmup and cosine decay learning rate schedule.

    Parameters:
        step (int): The current step in the learning rate schedule.
        warmup_steps (int): The number of steps for the linear warmup phase.
        max_steps (int): The total number of steps in the learning rate schedule.
        max_lr (float): The maximum learning rate.
        min_lr (float): The minimum learning rate.

    Returns:
        float: The learning rate for the given step.

    Raises:
        AssertionError: If the progress value is not between 0 and 1.

    The function implements a warmup and cosine decay learning rate schedule. The learning rate starts with a linear warmup phase, followed by a cosine decay phase. The warmup phase is defined by the `warmup_steps` parameter, and the cosine decay phase is defined by the `max_steps` and `max_lr` parameters. The minimum learning rate is defined by the `min_lr` parameter.

    The function calculates the progress value as the ratio of the current step to the total number of steps in the schedule. It then calculates the cosine decay coefficient using the progress value. Finally, it returns the learning rate by adding the minimum learning rate to the product of the cosine decay coefficient and the difference between the maximum learning rate and the minimum learning rate.

    Note: The function assumes that the `step` parameter is non-negative and less than or equal to the `max_steps` parameter.
    """
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
