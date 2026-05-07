"""Prepare tiny Shakespeare dataset into train.bin / val.bin.

~1M characters, ~300K GPT-2 BPE tokens.  Downloads automatically.

Usage:
    python data/shakespeare/prepare.py
    python data/shakespeare/prepare.py --out_dir=data/shakespeare --val_frac=0.1
"""

import argparse
import os
import urllib.request

import numpy as np
import tiktoken

DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def parse_args():
    p = argparse.ArgumentParser(description="Prepare tiny Shakespeare into train.bin/val.bin")
    p.add_argument("--out_dir", type=str, default=os.path.dirname(os.path.abspath(__file__)))
    p.add_argument("--val_frac", type=float, default=0.1)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # Download
    input_path = os.path.join(args.out_dir, "input.txt")
    if not os.path.exists(input_path):
        print(f"Downloading {DATA_URL} ...")
        urllib.request.urlretrieve(DATA_URL, input_path)

    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()
    print(f"Dataset: {len(text):,} characters")

    # Tokenize
    enc = tiktoken.get_encoding("gpt2")
    tokens = enc.encode_ordinary(text)
    tokens = np.array(tokens, dtype=np.uint16)
    print(f"Tokenized: {len(tokens):,} tokens")

    # Split
    split = int(len(tokens) * (1 - args.val_frac))
    train_tokens = tokens[:split]
    val_tokens = tokens[split:]

    # Write
    for name, arr in [("train", train_tokens), ("val", val_tokens)]:
        path = os.path.join(args.out_dir, f"{name}.bin")
        arr.tofile(path)
        print(f"{name}: {len(arr):,} tokens -> {path}")
