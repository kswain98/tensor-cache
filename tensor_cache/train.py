"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python tensor_cache/train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 tensor_cache/train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 tensor_cache/train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 tensor_cache/train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
import sys
import time
import math
import pickle
from contextlib import nullcontext
from pathlib import Path

# Make project root importable so `tensor_cache.*`, `utils.*`, `baselines.*` resolve
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from tensor_cache.model import GPTConfig, GPT
from utils import make_progress, cprint, ctprint, apply_overrides

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = 'out'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_at_start = True # run evaluation at iter 0
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
# wandb logging
wandb_log = False # disabled by default
wandb_project = 'owt'
wandb_run_name = 'gpt2' # 'run' + str(time.time())
wandb_entity = '' # optionally set your wandb entity/team
wandb_group = 'train'
wandb_log_train = True
wandb_log_system = True
wandb_log_grad_norm = True
# data
dataset = 'openwebtext'
gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024
# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
# adamw optimizer
learning_rate = 6e-4 # max learning rate
max_iters = 600000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
seed = 1337
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True # use PyTorch 2.0 to compile the model to be faster
compile_mode = 'default' # e.g. 'default', 'reduce-overhead', 'max-autotune' (torch>=2.1)
compile_disable_cudagraphs = False # keep torch.compile, but disable CUDAGraph capture paths
compile_clone_loss_for_backward = True # workaround for some cudagraph overwrite backward errors
experiment_suite = 'long-context'
wandb_reset_peak_each_iter = True
device_peak_tflops = 0.0 # optional hardware peak TFLOPS per GPU (e.g. H100/H200 bf16 ~989)

# Tensor Cache Config
# --- KV-free Tensor Cache flags (must exist before apply_overrides) ---
use_rope = True
rope_base = 10000

use_kv_cache = True
kv_window = 512  # 0 = unbounded full KV-cache baseline, >0 = fixed window
kv_num_sinks = 0  # attention sink tokens (StreamingLLM); 0 = disabled
kv_mode = 'window_kv'  # {'full_kv', 'window_kv', 'tc', 'streaming_llm', 'infini'}

tc_enabled = False
tc_layers = -1  # -1 => all layers, 0 => none, k>0 => top-k layers
tc_update_rule = "delta"      # "outer" or "delta"
tc_write_on_evict = True
tc_write_on_insert = False
tc_freeze_decay_lr = False    # freeze decay/lr (ablation: learned vs fixed)

tc_two_timescales = False
tc_chunk_size = 64
tc_decay_init = 0.995
tc_lr_init = 0.05
tc_gate_init = -2.0
tc_normalize_k = True
tc_normalize_q = False
tc_value_scale = 1.0

tc_decay_slow_init = 0.9995
tc_lr_slow_init = 0.01
tc_alpha_init = 0.5

tc_num_slots = 1
tc_read_topk = 1
tc_router_temp = 1.0
tc_write_route = 'k'  # {'k', 'q'}

tc_use_in_full_forward = True

# V2.2: per-token query-conditional fusion gate (instead of scalar per-layer gate)
tc_per_token_gate = False
# V2.3: read-time normalization (mLSTM/Infini-style z vector)
tc_normalize_read = False
tc_normalize_read_eps = 1.0

# Infini-attention baseline config
infini_enabled = False
infini_update_rule = "delta"    # "linear" or "delta"
infini_segment_size = 256  # must be < block_size for memory to be used during training



# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
apply_overrides(globals()) # overrides from command line or config file

if kv_mode == 'full_kv':
    use_kv_cache = True
    kv_window = 0
    tc_enabled = False
elif kv_mode == 'window_kv':
    use_kv_cache = True
    if kv_window <= 0:
        raise ValueError("kv_mode='window_kv' requires kv_window > 0.")
    tc_enabled = False
elif kv_mode == 'streaming_llm':
    use_kv_cache = True
    if kv_window <= 0:
        raise ValueError("kv_mode='streaming_llm' requires kv_window > 0.")
    tc_enabled = False
    if kv_num_sinks <= 0:
        kv_num_sinks = 4  # StreamingLLM needs attention sinks
elif kv_mode == 'infini':
    use_kv_cache = True
    if kv_window <= 0:
        raise ValueError("kv_mode='infini' requires kv_window > 0.")
    tc_enabled = False
    infini_enabled = True
elif kv_mode == 'tc':
    use_kv_cache = True
    if kv_window <= 0:
        raise ValueError("kv_mode='tc' requires kv_window > 0.")
    tc_enabled = True
    # Keep default eviction-write behavior unless caller explicitly overrides.
    # Guard against accidentally disabling all TC writes.
    if (not tc_write_on_evict) and (not tc_write_on_insert):
        cprint("warning: both tc_write_on_evict and tc_write_on_insert are False; enabling tc_write_on_evict")
        tc_write_on_evict = True
else:
    raise ValueError(f"Unknown kv_mode: {kv_mode}")

def build_model_args():
    return dict(
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        block_size=block_size,
        bias=bias,
        vocab_size=None,
        dropout=dropout,
        use_rope=use_rope,
        rope_base=rope_base,
        use_kv_cache=use_kv_cache,
        kv_window=kv_window,
        kv_num_sinks=kv_num_sinks,
        tc_enabled=tc_enabled,
        tc_layers=tc_layers,
        tc_update_rule=tc_update_rule,
        tc_write_on_evict=tc_write_on_evict,
        tc_write_on_insert=tc_write_on_insert,
        tc_freeze_decay_lr=tc_freeze_decay_lr,
        tc_two_timescales=tc_two_timescales,
        tc_chunk_size=tc_chunk_size,
        tc_decay_init=tc_decay_init,
        tc_lr_init=tc_lr_init,
        tc_gate_init=tc_gate_init,
        tc_normalize_k=tc_normalize_k,
        tc_normalize_q=tc_normalize_q,
        tc_value_scale=tc_value_scale,
        tc_decay_slow_init=tc_decay_slow_init,
        tc_lr_slow_init=tc_lr_slow_init,
        tc_alpha_init=tc_alpha_init,
        tc_num_slots=tc_num_slots,
        tc_read_topk=tc_read_topk,
        tc_router_temp=tc_router_temp,
        tc_write_route=tc_write_route,
        tc_use_in_full_forward=tc_use_in_full_forward,
        tc_per_token_gate=tc_per_token_gate,
        tc_normalize_read=tc_normalize_read,
        tc_normalize_read_eps=tc_normalize_read_eps,
        infini_enabled=infini_enabled,
        infini_update_rule=infini_update_rule,
        infini_segment_size=infini_segment_size,
    )

config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------


# various inits, derived attributes, I/O setup
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
cprint(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(seed + seed_offset)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# poor man's data loader
data_dir = os.path.join('data', dataset)
_data_memmaps = {}

def get_batch(split):
    # Keep one memmap per split per process to avoid per-step file open overhead.
    if split not in _data_memmaps:
        filename = 'train.bin' if split == 'train' else 'val.bin'
        _data_memmaps[split] = np.memmap(os.path.join(data_dir, filename), dtype=np.uint16, mode='r')
    data = _data_memmaps[split]
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    cprint(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# model init
model_args = build_model_args() # start with model_args from command line

if init_from == 'scratch':
    # init a new model from scratch
    cprint("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        cprint("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    cprint(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # Force all overlapping model args from checkpoint for strict state_dict compatibility.
    for k, v in checkpoint_model_args.items():
        if k in model_args:
            model_args[k] = v
    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
elif init_from.startswith('gpt2'):
    cprint(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    # initialize from OpenAI GPT-2 weights
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    # read off the created config params, so we can store them into checkpoint correctly
    for k in list(model_args.keys()):
        if hasattr(model.config, k):
            model_args[k] = getattr(model.config, k)
# crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size # so that the checkpoint will have the right value
model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
if hasattr(torch, 'amp') and hasattr(torch.amp, 'GradScaler'):
    scaler = torch.amp.GradScaler('cuda', enabled=(device_type == 'cuda' and dtype == 'float16'))
else:
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None # free up memory

# compile the model
if compile:
    cprint("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    if device_type == 'cuda' and compile_disable_cudagraphs:
        try:
            import torch._inductor.config as inductor_config
            if hasattr(inductor_config, "triton") and hasattr(inductor_config.triton, "cudagraphs"):
                inductor_config.triton.cudagraphs = False
            if hasattr(inductor_config, "cuda") and hasattr(inductor_config.cuda, "enable_cuda_graphs"):
                inductor_config.cuda.enable_cuda_graphs = False
            cprint("torch.compile cudagraphs disabled by config")
        except Exception as e:
            cprint(f"warning: could not set inductor cudagraph flags: {e}")
    try:
        model = torch.compile(model, mode=compile_mode) # requires PyTorch 2.0
    except TypeError:
        # Older torch.compile signatures do not expose mode=
        model = torch.compile(model)

# Some torch.compile configurations use CUDAGraph capture under the hood.
# Marking step boundaries avoids "accessing tensor output of CUDAGraphs that has
# been overwritten by a subsequent run" when outputs are consumed later in step.
def maybe_cudagraph_mark_step_begin():
    if not compile or device_type != 'cuda':
        return
    # Preferred API (PyTorch 2.3+)
    compiler_mod = getattr(torch, "compiler", None)
    if compiler_mod is not None:
        fn = getattr(compiler_mod, "cudagraph_mark_step_begin", None)
        if callable(fn):
            fn()
            return

    # Back-compat fallbacks used by older releases
    inductor_mod = getattr(torch, "_inductor", None)
    if inductor_mod is not None:
        fn = getattr(inductor_mod, "cudagraph_mark_step_begin", None)
        if callable(fn):
            fn()
            return
        trees_mod = getattr(inductor_mod, "cudagraph_trees", None)
        fn = getattr(trees_mod, "mark_step_begin", None) if trees_mod is not None else None
        if callable(fn):
            fn()

# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    eval_pbar = make_progress(
        total=2 * int(eval_iters),
        desc=f"eval@{iter_num}",
        position_offset=1,  # sit one line below the train bar instead of fighting for it
        leave=False,
    )
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            maybe_cudagraph_mark_step_begin()
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
            eval_pbar.update(1)
            eval_pbar.set_postfix(split=split, loss=f"{losses[:k+1].mean().item():.4f}")
        out[split] = losses.mean()
    eval_pbar.close()
    model.train()
    return out

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# logging
if wandb_log and master_process:
    import wandb
    wandb_init_kwargs = dict(project=wandb_project, name=wandb_run_name, group=wandb_group, config=config)
    if wandb_entity:
        wandb_init_kwargs["entity"] = wandb_entity
    wandb.init(**wandb_init_kwargs)

# training loop
X, Y = get_batch('train') # fetch the very first batch
t0 = time.time()
run_start_time = t0
raw_model = model.module if ddp else model # unwrap DDP container if needed
run_peak_alloc_gb = 0.0
run_peak_reserved_gb = 0.0

def estimate_model_compute_terms(model_obj):
    cfg_obj = model_obj.config
    n_params_total = int(model_obj.get_num_params(non_embedding=False))
    n_params_non_emb = int(model_obj.get_num_params(non_embedding=True))
    L = int(cfg_obj.n_layer)
    H = int(cfg_obj.n_head)
    Q = int(cfg_obj.n_embd // cfg_obj.n_head)
    T = int(cfg_obj.block_size)
    # PaLM-style rough FLOPs estimate for throughput/accounting.
    flops_per_token = float(6 * n_params_non_emb + 12 * L * H * Q * T)
    flops_per_fwdbwd = float(flops_per_token * T)
    return dict(
        n_params_total=n_params_total,
        n_params_non_emb=n_params_non_emb,
        flops_per_token=flops_per_token,
        flops_per_fwdbwd=flops_per_fwdbwd,
    )

fwdbwd_per_iter_local = int(batch_size * gradient_accumulation_steps)
compute_terms = estimate_model_compute_terms(raw_model)
flops_per_iter_local = float(compute_terms["flops_per_fwdbwd"] * fwdbwd_per_iter_local)
flops_per_iter_global = float(flops_per_iter_local * ddp_world_size)

if master_process:
    cprint(
        "compute estimate: "
        f"params_total={compute_terms['n_params_total']:,}, "
        f"params_non_emb={compute_terms['n_params_non_emb']:,}, "
        f"flops/token={compute_terms['flops_per_token']:.3e}, "
        f"flops/iter(local)={flops_per_iter_local:.3e}, "
        f"flops/iter(global)={flops_per_iter_global:.3e}"
    )
    if hasattr(raw_model, 'get_param_breakdown'):
        pb = raw_model.get_param_breakdown()
        cprint(
            f"param breakdown: base={pb['base']:,}, "
            f"tc_memory={pb['tc_memory']:,}, tc_proj={pb['tc_proj']:,}, "
            f"infini={pb['infini']:,}, overhead={pb['memory_overhead']:,} ({pb['overhead_pct']:.2f}%)"
        )
if wandb_log and master_process:
    wandb.run.summary["compute/params_total"] = int(compute_terms["n_params_total"])
    wandb.run.summary["compute/params_non_embedding"] = int(compute_terms["n_params_non_emb"])
    wandb.run.summary["compute/est_flops_per_token"] = float(compute_terms["flops_per_token"])
    wandb.run.summary["compute/est_flops_per_fwdbwd"] = float(compute_terms["flops_per_fwdbwd"])
    wandb.run.summary["compute/est_flops_per_iter_local"] = float(flops_per_iter_local)
    wandb.run.summary["compute/est_flops_per_iter_global"] = float(flops_per_iter_global)
    if hasattr(raw_model, 'get_param_breakdown'):
        pb = raw_model.get_param_breakdown()
        for k, v in pb.items():
            wandb.run.summary[f"params/{k}"] = v

train_pbar = None
postfix_state = {}
if master_process:
    train_total = int(max_iters) + 1
    train_pbar = make_progress(
        total=train_total,
        initial=min(int(iter_num), train_total),
        desc="train",
        leave=True,
    )

def _refresh_train_pbar():
    if train_pbar is not None and not getattr(train_pbar, "disable", False):
        train_pbar.set_postfix(**postfix_state)

while True:

    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # evaluate the loss on train/val sets and write checkpoints
    should_eval = (iter_num % eval_interval == 0) and (iter_num > 0 or eval_at_start)
    if should_eval and master_process:
        losses = estimate_loss()
        postfix_state["val"] = f"{losses['val']:.4f}"
        postfix_state["best"] = f"{best_val_loss:.4f}"
        _refresh_train_pbar()
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
            })
        if losses['val'] < best_val_loss or always_save_checkpoint:
            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                postfix_state["best"] = f"{best_val_loss:.4f}"
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
                postfix_state["ckpt"] = str(iter_num)
                _refresh_train_pbar()
    if iter_num == 0 and eval_only:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    if wandb_log and wandb_log_system and master_process and device_type == 'cuda' and wandb_reset_peak_each_iter:
        torch.cuda.reset_peak_memory_stats()
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        maybe_cudagraph_mark_step_begin()
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
        if compile and device_type == 'cuda' and compile_clone_loss_for_backward:
            loss = loss.clone()
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = get_batch('train')
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    # unscale gradients if needed for clipping and/or grad norm logging
    need_unscale = (grad_clip != 0.0) or (wandb_log and wandb_log_grad_norm and master_process)
    if need_unscale:
        scaler.unscale_(optimizer)

    grad_norm = None
    if grad_clip != 0.0:
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    elif wandb_log and wandb_log_grad_norm and master_process:
        grad_sq_sum = None
        for p in model.parameters():
            if p.grad is None:
                continue
            g = p.grad.detach()
            gs = torch.sum(g.float() * g.float())
            grad_sq_sum = gs if grad_sq_sum is None else (grad_sq_sum + gs)
        if grad_sq_sum is not None:
            grad_norm = torch.sqrt(grad_sq_sum)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    peak_alloc_gb = None
    peak_reserved_gb = None
    if wandb_log and wandb_log_system and master_process and device_type == 'cuda':
        # Update run peaks every iteration (not only when iter is logged),
        # so long log intervals do not under-report max memory.
        peak_alloc_gb = torch.cuda.max_memory_allocated() / (1024**3)
        peak_reserved_gb = torch.cuda.max_memory_reserved() / (1024**3)
        run_peak_alloc_gb = max(run_peak_alloc_gb, peak_alloc_gb)
        run_peak_reserved_gb = max(run_peak_reserved_gb, peak_reserved_gb)
    if iter_num % log_interval == 0 and master_process:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * gradient_accumulation_steps
        toks_per_s = tokens_per_iter / max(dt, 1e-12)
        postfix_state.update({
            "loss": f"{lossf:.4f}",
            "lr": f"{lr:.2e}",
            "tok_s": f"{toks_per_s:.0f}",
            "ms": f"{dt*1000:.0f}",
        })
        if train_pbar is not None and not getattr(train_pbar, "disable", False):
            train_pbar.set_postfix(**postfix_state)
        else:
            ctprint(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms")
        if wandb_log and wandb_log_train:
            est_tflops_local = (flops_per_iter_local / max(dt, 1e-12)) / 1e12
            est_tflops_global = est_tflops_local * ddp_world_size
            log_dict = {
                "iter": iter_num,
                "train/loss_step": lossf,
                "train/lr_step": lr,
                "train/iter_time_ms": dt * 1000.0,
                "train/tokens_per_s": toks_per_s,
                "train/tokens_per_iter": tokens_per_iter,
                "train/tokens_seen": (iter_num + 1) * tokens_per_iter,
                "compute/est_flops_per_token": float(compute_terms["flops_per_token"]),
                "compute/est_flops_per_iter_local": float(flops_per_iter_local),
                "compute/est_flops_per_iter_global": float(flops_per_iter_global),
                "compute/est_tflops_local": float(est_tflops_local),
                "compute/est_tflops_global": float(est_tflops_global),
            }
            if device_peak_tflops > 0.0:
                log_dict["compute/est_util_local_pct"] = float(100.0 * est_tflops_local / device_peak_tflops)
                log_dict["compute/est_util_global_pct"] = float(100.0 * est_tflops_global / (device_peak_tflops * ddp_world_size))
            if grad_norm is not None:
                log_dict["train/grad_norm"] = float(grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm)
            if dtype == 'float16':
                log_dict["train/grad_scale"] = float(scaler.get_scale())
            if wandb_log_system and device_type == 'cuda':
                if peak_alloc_gb is None:
                    peak_alloc_gb = torch.cuda.max_memory_allocated() / (1024**3)
                if peak_reserved_gb is None:
                    peak_reserved_gb = torch.cuda.max_memory_reserved() / (1024**3)
                log_dict.update({
                    "system/gpu_mem_alloc_gb": torch.cuda.memory_allocated() / (1024**3),
                    "system/gpu_mem_reserved_gb": torch.cuda.memory_reserved() / (1024**3),
                    "system/gpu_mem_peak_step_alloc_gb": peak_alloc_gb,
                    "system/gpu_mem_peak_step_reserved_gb": peak_reserved_gb,
                    "system/gpu_mem_peak_run_alloc_gb": run_peak_alloc_gb,
                    "system/gpu_mem_peak_run_reserved_gb": run_peak_reserved_gb,
                })
            wandb.log(log_dict)
    iter_num += 1
    if train_pbar is not None:
        train_pbar.update(1)

    # termination conditions
    if iter_num > max_iters:
        break

if train_pbar is not None:
    train_pbar.close()

if ddp:
    destroy_process_group()
if wandb_log and master_process:
    completed_updates = int(iter_num)
    run_wall_s = max(0.0, time.time() - run_start_time)
    run_gpu_hours = (run_wall_s / 3600.0) * float(ddp_world_size)
    total_est_flops_local = float(flops_per_iter_local * completed_updates)
    total_est_flops_global = float(flops_per_iter_global * completed_updates)
    wandb.run.summary["compute/completed_updates"] = completed_updates
    wandb.run.summary["compute/run_wall_s"] = run_wall_s
    wandb.run.summary["compute/run_gpu_hours"] = run_gpu_hours
    wandb.run.summary["compute/est_total_flops_local"] = total_est_flops_local
    wandb.run.summary["compute/est_total_flops_global"] = total_est_flops_global
    wandb.run.summary["compute/est_total_pflop_local"] = total_est_flops_local / 1e15
    wandb.run.summary["compute/est_total_pflop_global"] = total_est_flops_global / 1e15
    if wandb_log_system and device_type == 'cuda':
        wandb.run.summary["system/gpu_mem_peak_run_alloc_gb"] = run_peak_alloc_gb
        wandb.run.summary["system/gpu_mem_peak_run_reserved_gb"] = run_peak_reserved_gb
    wandb.run.summary["train/best_val_loss"] = best_val_loss
    wandb.finish()
