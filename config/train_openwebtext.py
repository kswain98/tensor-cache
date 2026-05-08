# Paper-scale config for training on OpenWebText (~9B tokens).
# This is the canonical run used in the paper: 124M-param GPT-2 architecture,
# block_size=1024, 600K iters. Expects multi-GPU and many hours/days of
# compute.
#
# Usage from repo root (single GPU):
#   python tensor_cache/train.py config/train_openwebtext.py
#
# Multi-GPU via DDP (recommended):
#   torchrun --standalone --nproc_per_node=8 tensor_cache/train.py \
#       config/train_openwebtext.py
#
# Override on top:
#   python tensor_cache/train.py config/train_openwebtext.py \
#       --kv_mode=window_kv --kv_window=512

# I/O
out_dir = 'checkpoints/openwebtext'
eval_interval = 2000
eval_iters = 200
log_interval = 1
eval_at_start = True
always_save_checkpoint = True

# wandb
wandb_log = False
wandb_project = 'owt'
wandb_run_name = 'tc'

# data
dataset = 'openwebtext'
gradient_accumulation_steps = 5 * 8  # 40 micro-batches per optim step
batch_size = 12
block_size = 1024

# 124M params (GPT-2 small)
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0

# optim
learning_rate = 6e-4
max_iters = 600000
lr_decay_iters = 600000
min_lr = 6e-5
warmup_iters = 2000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# tensor cache: paper-default 512-token window
kv_mode = 'tc'
kv_window = 512
tc_write_on_evict = True

compile = True
