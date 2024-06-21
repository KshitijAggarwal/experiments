import tiktoken
import torch


class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes):
        """
        Initializes the DataLoaderLite object with the provided batch size and sequence length.

        Parameters:
            B (int): The batch size for data loading.
            T (int): The sequence length for data loading.

        Returns:
            None
        """
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes

        tinyskp = "/Users/kshitijaggarwal/Documents/Projects/experiments/data/input.txt"
        with open(tinyskp, "r") as f:
            text = f.read()

        enc = tiktoken.get_encoding("gpt2")
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens, dtype=torch.long)
        print(f"Loaded {len(self.tokens)} tokens")
        print(f"1 Epoch will have {len(self.tokens) // (B * T)} batches")

        self.current_pos = (
            self.B * self.T * self.process_rank
        )  # striding out the processes

    def next_batch(self):
        """
        Returns the next batch of data from the DataLoaderLite object.

        This function retrieves the next batch of data from the DataLoaderLite object. It does this by slicing the `self.tokens` tensor to get the next batch of tokens. The batch size and sequence length are determined by the `B` and `T` attributes of the object, respectively.

        Parameters:
            None

        Returns:
            x (torch.Tensor): A tensor of shape `(B, T)` containing the input tokens for the batch.
            y (torch.Tensor): A tensor of shape `(B, T)` containing the target tokens for the batch.

        Note:
            - The function updates the `self.current_pos` attribute to keep track of the current position in the `self.tokens` tensor.
            - If the end of the `self.tokens` tensor is reached, the `self.current_pos` attribute is reset to 0.
        """
        B, T = self.B, self.T
        start = self.current_pos
        end = self.current_pos + B * T + 1
        buf = self.tokens[start:end]
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)

        self.current_pos += B * T * self.num_processes
        if self.current_pos + B * T * self.num_processes + 1 > len(self.tokens):
            self.current_pos = self.B * self.T * self.process_rank
        return x, y
