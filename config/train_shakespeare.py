# Tiny config for training on Shakespeare — runs end-to-end in ~1-2 minutes
# on a single GPU. Useful as a smoke test for any kv_mode / tc_* flag changes.
#
# Usage from repo root:
#   python tensor_cache/train.py config/train_shakespeare.py
#
# Override on top:
#   python tensor_cache/train.py config/train_shakespeare.py --kv_mode=window_kv

# I/O
out_dir = 'out-shakespeare'
eval_interval = 200
eval_iters = 20
log_interval = 10
eval_at_start = False
always_save_checkpoint = False  # only save when val loss improves

# wandb
wandb_log = False
wandb_project = 'shakespeare'
wandb_run_name = 'tc-mini'

# data
dataset = 'shakespeare'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 256

# tiny model (~5M params)
n_layer = 4
n_head = 4
n_embd = 128
dropout = 0.0

# optim
learning_rate = 1e-3
max_iters = 2000
lr_decay_iters = 2000
min_lr = 1e-4
warmup_iters = 100
beta2 = 0.99  # Karpathy's nanoGPT setting for tiny shakespeare

# tensor cache: small window so eviction kicks in even on short sequences
kv_mode = 'tc'
kv_window = 128
tc_write_on_evict = True

# torch.compile is mostly overhead for a 2K-iter run
compile = False
