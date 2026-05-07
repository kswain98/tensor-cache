# Saves the WikiText-2 (raw) dataset to train.bin / val.bin for training.

import argparse
import os

import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

enc = tiktoken.get_encoding("gpt2")


def parse_args():
    p = argparse.ArgumentParser(description="Prepare WikiText-2 (raw) into train.bin/val.bin")
    p.add_argument("--out_dir", type=str, default=os.path.dirname(__file__))
    p.add_argument("--num_proc", type=int, default=4)
    p.add_argument("--num_proc_load_dataset", type=int, default=4)
    p.add_argument("--total_batches", type=int, default=64)
    p.add_argument("--max_train_tokens", type=int, default=0, help="0 means no limit")
    p.add_argument("--max_val_tokens", type=int, default=0, help="0 means no limit")
    return p.parse_args()


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
    ids = enc.encode_ordinary(example["text"])
    ids.append(enc.eot_token)
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

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", num_proc=args.num_proc_load_dataset)

    # WikiText ships with train / validation / test. We use validation as 'val'.
    splits = {"train": ds["train"], "val": ds["validation"]}

    tokenized = {}
    for split, dset in splits.items():
        # WikiText has empty/whitespace-only rows; drop them so the EOT marker
        # actually delimits real documents.
        dset = dset.filter(lambda ex: ex["text"].strip() != "", num_proc=args.num_proc)
        tokenized[split] = dset.map(
            process,
            remove_columns=["text"],
            desc=f"tokenizing {split}",
            num_proc=args.num_proc,
        )

    tokenized["train"] = maybe_limit_tokens(tokenized["train"], args.max_train_tokens)
    tokenized["val"] = maybe_limit_tokens(tokenized["val"], args.max_val_tokens)

    for split, dset in tokenized.items():
        filename = os.path.join(args.out_dir, f"{split}.bin")
        n_tokens = write_split_bin(dset, filename, args.total_batches)
        print(f"{split}: docs={len(dset)} tokens={n_tokens} -> {filename}")
