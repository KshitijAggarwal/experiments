import os
import numpy as np
import tiktoken
from datasets import load_dataset
from multiprocessing import Pool, cpu_count


def tokenize(text, eot, enc):
    tokens = [eot] + enc.encode(text)
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (
        tokens_np < 2**16
    ).all(), "token dictionary too large for uint16"
    tokens_np_uint16 = tokens_np.astype(np.uint16)
    return tokens_np_uint16


def process_chunk(chunk_id):
    enc = tiktoken.get_encoding("gpt2")
    eot = enc._special_tokens["<|endoftext|>"]

    chunk_size = 100_000_000  # 100M tokens per file
    all_tokens = np.empty((chunk_size,), dtype=np.uint16)
    token_count = 0

    fw = load_dataset(
        "HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True
    )
    fw = fw.skip(chunk_id * chunk_size)

    for example in fw:
        text = example["text"]
        tokens = tokenize(text, eot, enc)
        if token_count + len(tokens) < chunk_size:
            all_tokens[token_count : token_count + len(tokens)] = tokens
            token_count += len(tokens)
        else:
            break

    all_tokens = all_tokens[:token_count]

    local_dir = "edu_fineweb10B"
    data_dir_path = ""
    os.makedirs(data_dir_path + local_dir, exist_ok=True)

    filename = os.path.join(
        data_dir_path + local_dir, f"sample-100MT-{chunk_id:03d}.npy"
    )
    np.save(filename, all_tokens)

    return f"Chunk {chunk_id} processed and saved."


if __name__ == "__main__":
    num_chunks = 100
    num_cpus = min(128, cpu_count())  # Use all available CPUs, up to 128

    with Pool(num_cpus) as p:
        results = p.map(process_chunk, range(num_chunks))

    for result in results:
        print(result)
