# Config for training on a PG-19 slice (~15M training tokens of long-form
# Project Gutenberg books). PG-19 is the standard long-context LM benchmark
# (Compressive Transformer; Infini-attention) — books are ~1M tokens each, so
# far-context modeling actually matters, unlike WikiText-2's short articles.
#
# Build the data first (offline from the HF cache):
#   python data/pg19/prepare.py
#
# Usage from repo root:
#   python tensor_cache/train.py config/train_pg19.py
#
# Override on top (e.g. to compare methods):
#   python tensor_cache/train.py config/train_pg19.py --kv_mode=streaming_llm

# I/O
out_dir = 'checkpoints/pg19'
eval_interval = 1000
eval_iters = 100
log_interval = 50
eval_at_start = False
always_save_checkpoint = False  # only save when val loss improves

# wandb
wandb_log = False
wandb_project = 'pg19'
wandb_run_name = 'tc-mid'

# data
dataset = 'pg19'
gradient_accumulation_steps = 2
batch_size = 32
block_size = 512

# ~16M params
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
