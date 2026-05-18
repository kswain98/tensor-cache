# Saves a slice of PG-19 (long-form Project Gutenberg books) to
# train.bin / val.bin for training. PG-19 is the standard long-context
# language-modeling benchmark (Compressive Transformer; Infini-attention).
# Books are ~1M tokens each, so a coherent long stream where far-context
# modeling actually matters — unlike WikiText-2's short articles.
#
# PG-19 is huge; we stream the first books until the token budget is met.

import argparse
import os
import pickle

import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

enc = tiktoken.get_encoding("gpt2")


def parse_args():
    p = argparse.ArgumentParser(description="Prepare a PG-19 slice into train.bin/val.bin")
    p.add_argument("--out_dir", type=str, default=os.path.dirname(__file__))
    p.add_argument("--hf_name", type=str, default="emozilla/pg19",
                   help="HF dataset id (emozilla/pg19 mirror works offline from cache)")
    p.add_argument("--max_train_tokens", type=int, default=15_000_000)
    p.add_argument("--max_val_tokens", type=int, default=2_000_000)
    return p.parse_args()


def fill_split(stream_iter, token_budget, desc):
    """Pull whole books from the stream, tokenize, until token_budget reached."""
    chunks, total = [], 0
    pbar = tqdm(total=token_budget, desc=desc, unit="tok")
    for rec in stream_iter:
        text = rec.get("text", "")
        if not text.strip():
            continue
        ids = enc.encode_ordinary(text)
        ids.append(enc.eot_token)  # delimit books
        chunks.append(np.asarray(ids, dtype=np.uint16))
        total += len(ids)
        pbar.update(len(ids))
        if total >= token_budget:
            break
    pbar.close()
    return np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.uint16)


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    ds = load_dataset(args.hf_name, split="train", streaming=True)
    it = iter(ds)

    train_arr = fill_split(it, args.max_train_tokens, "train")
    val_arr = fill_split(it, args.max_val_tokens, "val")  # disjoint books (stream continues)

    for name, arr in (("train", train_arr), ("val", val_arr)):
        if arr.size == 0:
            raise RuntimeError(f"{name} split empty — dataset stream exhausted?")
        path = os.path.join(args.out_dir, f"{name}.bin")
        m = np.memmap(path, dtype=np.uint16, mode="w+", shape=(arr.size,))
        m[:] = arr
        m.flush()
        print(f"{name}: {arr.size:,} tokens -> {path}")

    with open(os.path.join(args.out_dir, "meta.pkl"), "wb") as f:
        pickle.dump({"vocab_size": 50304, "dataset": "pg19",
                     "source": args.hf_name}, f)
    print("vocab_size: 50304 (GPT-2 BPE)")
