# Config for training on a WikiText-103 slice (~82M training tokens).
# WikiText-103 is the standard mid-scale long-context LM benchmark. The large
# training set means a ~31M model does NOT overfit at a few-thousand iters, so
# (unlike WikiText-2 / Shakespeare) windowed-training does not act as a
# regularizer — making this the cleanest setting to isolate the TC mechanism.
#
# Build the data first (offline from the HF cache):
#   python data/wikitext103/prepare.py --max_train_docs=800000
#
# Usage from repo root:
#   python tensor_cache/train.py config/train_wikitext103.py
#
# Override on top (e.g. to compare methods):
#   python tensor_cache/train.py config/train_wikitext103.py --kv_mode=streaming_llm

# I/O
out_dir = 'checkpoints/wikitext103'
eval_interval = 1000
eval_iters = 100
log_interval = 50
eval_at_start = False
always_save_checkpoint = False  # only save when val loss improves

# wandb
wandb_log = False
wandb_project = 'wikitext103'
wandb_run_name = 'tc-mid'

# data
dataset = 'wikitext103'
gradient_accumulation_steps = 1
batch_size = 32
block_size = 512

# ~31M params (same as the PG-19 setup, for direct comparability)
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.1

# optim
learning_rate = 3e-4
max_iters = 12000
lr_decay_iters = 12000
min_lr = 3e-5
warmup_iters = 200
beta2 = 0.95

# tensor cache
kv_mode = 'tc'
kv_window = 256
tc_write_on_evict = True

compile = True
