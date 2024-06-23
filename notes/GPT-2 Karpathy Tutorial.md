# Notes from GPT-2 Tutorial

Video: https://www.youtube.com/watch?v=l8pRSuU81PU

```python
# creates an interative shell at that point in the code executon

# code code
import code; code.interact(local=locals())
# more code

```


He explains kernel fusion really well. 

Operator Fusion: Allows you to move data to chip (GPU) from the HBM (High bandwidth memory), perform lots of operations on it (on chip), and then move it back to HBM. Without it, every operation would require a data transfer from chip to the HBM. 

FlashAttention: Fused Attention kernel. torch.complie() can't do it. 
	Uses online sofmax trick, which evaluates softmax without seeing all the numbers, does it online.
Memory access patterns is more imp than total flops!!

Nice numbers vs not-so-nice numbers. Use nice numbers in code. Helps with CUDA. 32 is nice. 17 isn't nice. Nice numbers are divisible by 2,4,8,16,32,64,128 .. 


# Doing it.. 

```python 
self.choices = nn.ModuleDict({
		'conv': nn.Conv2d(10, 10, 3),
		'pool': nn.MaxPool2d(3)
})
```

```python
from dataclasses import dataclass

@dataclass
class GPTConfig:
	vocab_size: int = 10
	n_embed: int = 100
	max_seq_length: int = 100

config = GPTConfig()

```

```python

torch.nn.LayerNorm(_normalized_shape_)

```

Size of Linear/FF in MLP is (n_embed x 4 * n_embed). 

==This 4 * n_embed seems arbitrary. Was there is the original AIAYN paper too (Section 3.3). Has this been ablated? What is it called? Why 4?==


# Attention Mechanism
From other tutorial: https://www.youtube.com/watch?v=kCc8FmEb1nY
Look at 

```python

# qkv.shape -> B, T, 3 * n_embed
# Dividing qkv into chunks of size n_embed 
# based on axis number 2
q, k, v = qkv.split(n_embed, dim=2) 

```

```python
# Used for attention calculation. This enables the matrix multiplication trick!
torch.tril(torch.ones(3, 3))

# tensor([[1., 0., 0.],
#         [1., 1., 0.],
#         [1., 1., 1.]])

```

```python 

wei = torch.tril(torch.ones(T, T))
wei = wei / torch.sum(wei, dim=1, keepdim=True)

# x shape -> B, T, C
# wei shape -> T, T
# wei @ x -> pytorch will add a batch dim to wei
# so, (B, T, T) @ (B, T, C)
# @ is a batched matrix multiplication
# so B (T, T) x (T, C) -> (B, T, C)

xbow2 = wei @ x

```

```python 

torch.transpose(_input_, _dim0_, _dim1_)
# Returns a tensor that is a transposed version of `input`. 
# The given dimensions `dim0` and `dim1` are swapped.

```

### Encoder vs Decoder attention block

in a decoder block you have: 
```python 
wei = wei.masked_fill(tril == 0, float("-inf")) # setting future to -inf
```
So that tokens can't see future. This is the autoregressive nature. 

But in encoder block, this isn't there. So all tokens can attend to all tokens (even future ones), and so it can do things like infilling, etc. 


### Self Attention Minimal Code Example

```python 
import torch
import torch.nn.functional as F
import torch.nn as nn


torch.manual_seed(42)
B, T, C = 4, 8, 32 # Batch, time, channels
x = torch.randn(B, T, C)

# single head of self attention
head_size = 16
key = nn.Linear(C, head_size, bias=False)
query = nn.Linear(C, head_size, bias=False)
value = nn.Linear(C, head_size, bias=False)

k = key(x) # (B, T, head_size)
q = query(x) # (B, T, head_size)

# no communication between tokens happened yet
wei = q @ k.transpose(1, 2) * C ** (-0.5)
# (B, T, T) # not tokens communicate with each other to get affinities
# C ** (-0.5) is for scaling so that variance of wei is close to 1.
# otherwise after softmax it can get too peaky and not diffuse, esp at initialization

# wei = torch.zeros((T, T)) # tokens are independent/uniform. No affinities between tokens

tril = torch.tril(torch.ones((T, T))) # only look at past tokens
wei = wei.masked_fill(tril == 0, float("-inf")) # setting future to -inf
wei = F.softmax(wei, dim=-1) # softmax to get weights for averaging along time

v = value(x) # (B, T, head_size)
out = wei @ v # actual weighted sum or average along time
out.shape # B, T, head_size

```

```python
torch.unsqueeze(_input_, _dim_) → [Tensor]

# Returns a new tensor with a dimension of size one inserted at the specified position. The returned tensor shares the same underlying data with this tensor.

# Example:

>>> x = torch.tensor([1, 2, 3, 4])
>>> torch.unsqueeze(x, 0)
tensor([[ 1,  2,  3,  4]])
>>> torch.unsqueeze(x, 1)
tensor([[ 1],
        [ 2],
        [ 3],
        [ 4]])
```


```python

torch.multinomial(_input_, _num_samples_, _replacement=False_, _*_, _generator=None_, _out=None_) → LongTensor[]

# Returns a tensor where each row contains `num_samples` indices sampled from the multinomial (a stricter definition would be multivariate, refer to torch.distributions.multinomial.Multinomial for more details) probability distribution located in the corresponding row of tensor `input`.

torch.topk(_input_, _k_, _dim=None_, _largest=True_, _sorted=True_, _*_, _out=None_)
# Returns the `k` largest elements of the given `input` tensor along a given dimension.

torch.gather(_input_, _dim_, _index_, _*_, _sparse_grad=False_, _out=None_) → [Tensor]

# Gathers values along an axis specified by dim.
```
### nn.Embedding

A simple lookup table that stores embeddings of a fixed dictionary and size.

`nn.Embedding` is a PyTorch module that represents an embedding layer in a neural network. It is commonly used to map discrete input tokens (such as words or token indices) to dense vector representations (embeddings) in a continuous vector space.

The `nn.Embedding` layer is initialized with a fixed vocabulary size (`num_embeddings`) and an embedding dimension (`embedding_dim`). It maintains a learnable weight matrix of shape `(num_embeddings, embedding_dim)`, where each row corresponds to the embedding vector for a specific token.

During the forward pass, the `nn.Embedding` layer takes in a tensor of token indices and retrieves the corresponding embedding vectors from the weight matrix using indexing. The resulting output is a tensor of shape `(batch_size, sequence_length, embedding_dim)`, where `batch_size` is the number of input sequences and `sequence_length` is the length of each input sequence.

The embedding layer is trained along with the rest of the neural network using backpropagation and gradient descent. During training, the gradients are computed with respect to the loss function, and the embedding weights are updated accordingly. The embedding layer learns to map the input tokens to meaningful representations that capture semantic and syntactic information relevant to the task at hand.

```python
y -> Shape: A, B
y.view(-1) -> Shape: A*B

```

### Initialization
```python 
	self.apply(self._init_weights)
	
def _init_weights(self, module):
	if isinstance(module, nn.Linear):
		torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
		if module.bias is not None:
			torch.nn.init.zeros_(module.bias)
	elif isinstance(module, nn.Embedding):
		torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
		
```

### Var of residual stream grows from all the layers

the signal in residual stream keeps getting added. N times. 
Think.. N random numbers added together will have a std of N^(0.5). 
So we have to scale it down by this factor to keep the std to 1.

The two residual streams per layer. 
```python
class Block():
...
x = x + self.attn(self.ln_1(x))
x = x + self.mlp(self.ln_2(x))

class MLP:
	def forward(self, x):
		x = self.c_fc(x)
		x = self.gelu(x)
		x = self.c_proj(x)
		return x

# so we scale the weights of c_proj by 1/ (2 * nlayers) ** 0.5
...
```

gradient clipping, GPT-3 paper.
```python
norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
```
to prevent too high gradient due to bad batch, etc. to prevent model to get too big of a shock!
(kinda hacky but works)

### Learning Rate Scheduler
cosine decay lr scheduler used in GPT-3

### Weight decay:
It is kind of a regularization. Forcing the network to distribute the work across weights. 

it is common to not weight decay Biases and any other 1D tensors (layenorm, scales and biases). 

### Gradient Accumulation
If I want to do a batch size that's larger than what my GPU can hold. Gradient acc let's you do this sequentially to simulate that batch size. 

> loss.backward() has an inherent += in it. 

Gradient Accumulation: When you call `loss.backward()`, PyTorch doesn't just compute the gradients - it accumulates them. This means that if a parameter already has a gradient stored, the newly computed gradient is added to the existing one, rather than replacing it. 
`p.grad += (newly computed gradient)`

This behavior is crucial for scenarios where you want to accumulate gradients over multiple forward and backward passes before performing a single optimization step.
	
### DDP 
Imagine that there will be 8 (or N) processes running in parallel.

```python 

if torch.cuda.is_available():
	backend = "nccl" # need nccl for cuda + DDP
else:
	backend = "gloo" # for ddp on cpu

init_process_group(backend=backend)
ddp_rank = int(os.environ("RANK")) # uid for all processes across all nodes
ddp_local_rank = int(os.environ("LOCAL_RANK")) # uid for all processes on a node
ddp_world_size = int(os.environ("WORLD_SIZE")) # total no. of processes running

```

## Mistakes: 

All in CausalSelfAttention. 

Accidentally did: 
```python
q = k.view(B, T, n_heads, n_embed // n_heads).transpose(1, 2) # B, T, nh, head_size -> B, nh, T, head_size

# also did
attn = (q @ k.transpose(-1, -2) * n_embed ** (-0.5)) # (B, nh, T, head_size) x (B, nh, head_size, T) -> (B, nh, T, T)

# But that's not right as we have multiple heads. So normalization should be over head size. Correct is: 

attn = (q @ k.transpose(-1, -2) * k.size(-1) ** (-0.5)) 
# (B, nh, T, head_size) x (B, nh, head_size, T) -> (B, nh, T, T)

```

