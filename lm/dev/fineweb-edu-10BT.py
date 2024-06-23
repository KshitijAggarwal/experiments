import os
import numpy as np
import tiktoken
from datasets import load_dataset, config
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import mmap
import contextlib

# Set custom cache directory on the network volume
network_volume = "/workspace"
cache_dir = os.path.join(network_volume, ".cache/huggingface/datasets")
os.environ['HF_DATASETS_CACHE'] = cache_dir
config.HF_DATASETS_CACHE = cache_dir

# Ensure the cache directory exists
os.makedirs(cache_dir, exist_ok=True)

def tokenize(text, eot, enc):
    tokens = [eot] + enc.encode(text)
    return np.array(tokens, dtype=np.uint16)

def process_chunk(args):
    chunk_id, start_idx, end_idx = args
    enc = tiktoken.get_encoding("gpt2")
    eot = enc._special_tokens["<|endoftext|>"]
    
    chunk_size = end_idx - start_idx
    all_tokens = np.empty((chunk_size,), dtype=np.uint16)
    token_count = 0
    
    # Use streaming to load only the required portion of the dataset
    dataset = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
    dataset = dataset.skip(start_idx)
    
    for i, example in enumerate(dataset):
        if i >= chunk_size:
            break
        
        text = example["text"]
        tokens = tokenize(text, eot, enc)
        if token_count + len(tokens) <= chunk_size:
            all_tokens[token_count : token_count + len(tokens)] = tokens
            token_count += len(tokens)
        else:
            break
    
    all_tokens = all_tokens[:token_count]
    
    data_dir = os.path.join(network_volume, "edu_fineweb10B")
    os.makedirs(data_dir, exist_ok=True)
    filename = os.path.join(data_dir, f"sample-100MT-{chunk_id:03d}.npy")
    np.save(filename, all_tokens)
    
    return f"Chunk {chunk_id} processed and saved. Tokens: {token_count}"

def main():
    num_chunks = 100
    num_cpus = cpu_count() // 2
    total_tokens = 10_000_000_000  # 10B tokens
    tokens_per_chunk = total_tokens // num_chunks
    
    print(f"Processing {num_chunks} chunks using {num_cpus} CPUs...")
    with Pool(num_cpus) as p:
        args = [(i, i*tokens_per_chunk, (i+1)*tokens_per_chunk) for i in range(num_chunks)]
        results = list(tqdm(p.imap(process_chunk, args), total=num_chunks))
    
    for result in results:
        print(result)

    print("All chunks processed.")

if __name__ == "__main__":
    main()