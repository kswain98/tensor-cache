# saves the openwebtext dataset to a binary file for training. following was helpful:
# https://github.com/HazyResearch/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py

import argparse
import os

import numpy as np
import tiktoken
from datasets import load_dataset # huggingface datasets
from tqdm import tqdm

enc = tiktoken.get_encoding("gpt2")


def parse_args():
    p = argparse.ArgumentParser(description="Prepare OpenWebText into train.bin/val.bin")
    p.add_argument("--out_dir", type=str, default=os.path.dirname(__file__))
    p.add_argument("--num_proc", type=int, default=8)
    p.add_argument("--num_proc_load_dataset", type=int, default=8)
    p.add_argument("--val_frac", type=float, default=0.0005)
    p.add_argument("--split_seed", type=int, default=2357)
    p.add_argument("--total_batches", type=int, default=1024)
    p.add_argument("--train_docs_limit", type=int, default=0, help="0 means no limit")
    p.add_argument("--val_docs_limit", type=int, default=0, help="0 means no limit")
    p.add_argument("--max_train_tokens", type=int, default=0, help="0 means no limit")
    p.add_argument("--max_val_tokens", type=int, default=0, help="0 means no limit")
    return p.parse_args()


def maybe_limit_docs(dset, docs_limit):
    if docs_limit and docs_limit > 0 and docs_limit < len(dset):
        return dset.select(range(docs_limit))
    return dset


def maybe_limit_tokens(dset, token_limit):
    if not token_limit or token_limit <= 0:
        return dset
    lengths = np.asarray(dset["len"], dtype=np.int64)
    csum = np.cumsum(lengths)
    keep = int(np.searchsorted(csum, token_limit, side="right"))
    if keep <= 0:
        keep = 1
    if keep < len(dset):
        return dset.select(range(keep))
    return dset


def process(example):
    ids = enc.encode_ordinary(example["text"]) # encode_ordinary ignores special tokens
    ids.append(enc.eot_token) # add end of text token (50256 for GPT-2 BPE)
    return {"ids": ids, "len": len(ids)}


def write_split_bin(dset, filename, total_batches):
    if len(dset) == 0:
        raise ValueError(f"Split is empty after filtering; cannot write {filename}")
    arr_len = np.sum(dset["len"], dtype=np.uint64)
    arr = np.memmap(filename, dtype=np.uint16, mode="w+", shape=(arr_len,))
    n_shards = max(1, min(int(total_batches), int(len(dset))))
    idx = 0
    for batch_idx in tqdm(range(n_shards), desc=f"writing {filename}"):
        batch = dset.shard(num_shards=n_shards, index=batch_idx, contiguous=True).with_format("numpy")
        arr_batch = np.concatenate(batch["ids"])
        arr[idx: idx + len(arr_batch)] = arr_batch
        idx += len(arr_batch)
    arr.flush()
    return int(arr_len)


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # full OWT is large (~8M docs, ~9B tokens); these limits support quick experiments.
    dataset = load_dataset("openwebtext", num_proc=args.num_proc_load_dataset)
    split_dataset = dataset["train"].train_test_split(
        test_size=args.val_frac,
        seed=args.split_seed,
        shuffle=True,
    )
    split_dataset["val"] = split_dataset.pop("test")

    tokenized = split_dataset.map(
        process,
        remove_columns=["text"],
        desc="tokenizing the splits",
        num_proc=args.num_proc,
    )

    tokenized["train"] = maybe_limit_docs(tokenized["train"], args.train_docs_limit)
    tokenized["val"] = maybe_limit_docs(tokenized["val"], args.val_docs_limit)
    tokenized["train"] = maybe_limit_tokens(tokenized["train"], args.max_train_tokens)
    tokenized["val"] = maybe_limit_tokens(tokenized["val"], args.max_val_tokens)

    for split, dset in tokenized.items():
        filename = os.path.join(args.out_dir, f"{split}.bin")
        n_tokens = write_split_bin(dset, filename, args.total_batches)
        print(f"{split}: docs={len(dset)} tokens={n_tokens} -> {filename}")
