# Mid-size config for training on WikiText-2 (~2M training tokens).
# Sized between Shakespeare (toy) and OpenWebText (paper). Trains in ~10-20 min
# on a single GPU.
#
# Usage from repo root:
#   python tensor_cache/train.py config/train_wikitext2.py
#
# Override on top:
#   python tensor_cache/train.py config/train_wikitext2.py --kv_mode=window_kv

# I/O
out_dir = 'checkpoints/wikitext2'
eval_interval = 500
eval_iters = 50
log_interval = 50
eval_at_start = False
always_save_checkpoint = False  # only save when val loss improves

# wandb
wandb_log = False
wandb_project = 'wikitext2'
wandb_run_name = 'tc-mid'

# data
dataset = 'wikitext2'
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
max_iters = 20000
lr_decay_iters = 20000
min_lr = 3e-5
warmup_iters = 200
beta2 = 0.95

# tensor cache
kv_mode = 'tc'
kv_window = 256
tc_write_on_evict = True

compile = True
