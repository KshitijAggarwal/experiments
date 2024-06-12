def print_model_summary(model):
    """
    A function that prints the summary of the model including layer names, output shapes, and parameter numbers.

    Parameters:
    model (torch.nn.Module): The model for which the summary is printed.

    Returns:
    None
    """
    print("Model Summary:")
    print("-" * 50)
    print("{:>20}  {:>25} {:>15}".format("Layer (type)", "Output Shape", "Param #"))
    print("=" * 50)

    total_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            num_params = param.numel()
            total_params += num_params
            print("{:>20}  {:>25} {:>15}".format(name, str(param.shape), num_params))

    print("=" * 50)
    print(f"Total Trainable Params: {total_params}")
    print("-" * 50)
