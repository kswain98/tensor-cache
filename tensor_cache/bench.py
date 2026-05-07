"""
Benchmarking script for TensorCache inference modes.

Benchmarks memory usage and throughput across inference modes:
  - full_kv            (unbounded KV cache)
  - window_kv          (sliding window KV)
  - streaming_llm      (attention sinks + sliding window)
  - infini             (Infini-attention: sliding window + compressive memory)
  - tc                 (TensorCache: sliding window + tensor cache)
"""

import csv
import os
import time
from contextlib import nullcontext
from datetime import datetime, timezone

import numpy as np
import torch

from model import GPT, GPTConfig
from utils import make_progress, cprint, ctprint, apply_overrides

# -----------------------------------------------------------------------------
# benchmark modes
# one mode example: bench_modes = 'tc'
# all baseline modes: bench_modes = 'full_kv,window_kv,streaming_llm,infini,tc'
bench_modes = 'full_kv,window_kv,tc'

# model size
batch_size = 12
block_size = 1024
vocab_size = 50304
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0
bias = False

# data
real_data = True
dataset = 'openwebtext'

# tensor-cache / kv settings
kv_window = 512
use_rope = True
rope_base = 10000
tc_layers = -1
tc_update_rule = "delta"
tc_write_on_evict = True
tc_write_on_insert = False
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

# Infini-attention baseline config
infini_update_rule = "delta"
infini_segment_size = 256  # must be < block_size for memory to be used during training

# StreamingLLM baseline config
kv_num_sinks = 4

# optimizer
weight_decay = 1e-2
learning_rate = 1e-4
beta1 = 0.9
beta2 = 0.95

# runtime
seed = 1337
device = 'cuda'  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'  # 'float32'/'bfloat16'/'float16'
compile = True
compile_mode = 'default'  # e.g. 'default', 'reduce-overhead', 'max-autotune' (torch>=2.1)
compile_disable_cudagraphs = False
profile = False  # for bench_task='train': use pytorch profiler, or simple benchmarking
burnin_steps = 10
bench_steps = 20

# bench task
# - "train": original train-step throughput benchmark
# - "long_context": prefill/OOM sweep across prompt lengths
bench_task = 'train'  # {'train', 'long_context'}
long_context_lengths = '2048,4096,8192,16384,32768,65536,131072'
long_context_batch_size = 1
long_context_use_stream = True
long_context_print_errors = False
bench_decode = True
decode_steps = 128
output_csv = ''

# experiment metadata / logging
experiment_suite = 'long-context'
device_peak_tflops = 0.0  # optional peak TFLOPS per GPU for utilization estimate
wandb_log = False
wandb_project = 'owt'
wandb_run_name = 'bench'
wandb_entity = ''
wandb_group = 'bench'
apply_overrides(globals())  # overrides from command line or config file
# -----------------------------------------------------------------------------


def parse_modes(modes_str: str):
    modes = [m.strip() for m in modes_str.split(',') if m.strip()]
    allowed = {'full_kv', 'window_kv', 'streaming_llm', 'infini', 'tc'}
    bad = [m for m in modes if m not in allowed]
    if bad:
        raise ValueError(f"Unknown modes: {bad}. Allowed: {sorted(allowed)}")
    return modes

def parse_int_csv(values: str):
    out = []
    for s in values.split(','):
        s = s.strip()
        if not s:
            continue
        out.append(int(s))
    if not out:
        raise ValueError("Expected at least one integer value.")
    return out


def csv_append(path, columns, row_dict):
    """Append a single row to a CSV file, creating with header if needed."""
    if not path:
        return
    write_header = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=columns)
        if write_header:
            w.writeheader()
        w.writerow(row_dict)


def build_model(mode: str):
    common = dict(
        block_size=block_size,
        vocab_size=vocab_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        dropout=dropout,
        bias=bias,
    )
    if mode == 'full_kv':
        use_kv_cache = True
        kv_w = 0
        tc_enabled = False
        infini_enabled = False
    elif mode == 'window_kv':
        if kv_window <= 0:
            raise ValueError("window_kv mode requires kv_window > 0.")
        use_kv_cache = True
        kv_w = kv_window
        tc_enabled = False
        infini_enabled = False
    elif mode == 'streaming_llm':
        if kv_window <= 0:
            raise ValueError("streaming_llm mode requires kv_window > 0.")
        use_kv_cache = True
        kv_w = kv_window
        tc_enabled = False
        infini_enabled = False
    elif mode == 'infini':
        if kv_window <= 0:
            raise ValueError("infini mode requires kv_window > 0.")
        use_kv_cache = True
        kv_w = kv_window
        tc_enabled = False
        infini_enabled = True
    elif mode == 'tc':
        if kv_window <= 0:
            raise ValueError("tc mode requires kv_window > 0.")
        use_kv_cache = True
        kv_w = kv_window
        tc_enabled = True
        infini_enabled = False
    else:
        raise ValueError(f"Unknown mode: {mode}")

    tc_cfg = dict(
        use_rope=use_rope,
        rope_base=rope_base,
        use_kv_cache=use_kv_cache,
        kv_window=kv_w,
        kv_num_sinks=kv_num_sinks if mode == 'streaming_llm' else 0,
        tc_enabled=tc_enabled,
        tc_layers=tc_layers,
        tc_update_rule=tc_update_rule,
        tc_write_on_evict=tc_write_on_evict if tc_enabled else False,
        tc_write_on_insert=tc_write_on_insert,
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
        infini_enabled=infini_enabled,
        infini_update_rule=infini_update_rule,
        infini_segment_size=infini_segment_size,
    )
    return GPT(GPTConfig(**common, **tc_cfg))


def maybe_compile_model(model):
    if not compile:
        return model
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
        return torch.compile(model, mode=compile_mode)
    except TypeError:
        return torch.compile(model)

def sync_if_cuda():
    if device_type == 'cuda':
        torch.cuda.synchronize()

def reset_peak_if_cuda():
    if device_type == 'cuda':
        torch.cuda.reset_peak_memory_stats()

def peak_mem_gb():
    if device_type != 'cuda':
        return float('nan')
    return torch.cuda.max_memory_allocated() / (1024 ** 3)

def peak_reserved_mem_gb():
    if device_type != 'cuda':
        return float('nan')
    return torch.cuda.max_memory_reserved() / (1024 ** 3)

def estimate_model_compute_terms(model_obj):
    cfg_obj = model_obj.config
    n_params_total = int(model_obj.get_num_params(non_embedding=False))
    n_params_non_emb = int(model_obj.get_num_params(non_embedding=True))
    L = int(cfg_obj.n_layer)
    H = int(cfg_obj.n_head)
    Q = int(cfg_obj.n_embd // cfg_obj.n_head)
    T = int(cfg_obj.block_size)
    flops_per_token = float(6 * n_params_non_emb + 12 * L * H * Q * T)
    flops_per_fwdbwd = float(flops_per_token * T)
    return dict(
        n_params_total=n_params_total,
        n_params_non_emb=n_params_non_emb,
        flops_per_token=flops_per_token,
        flops_per_fwdbwd=flops_per_fwdbwd,
    )


torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
modes = parse_modes(bench_modes)

# data loading init
if real_data:
    data_dir = os.path.join('data', dataset)
    train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')

    def get_batch(split):
        data = train_data  # benchmark script ignores split
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([torch.from_numpy((data[i:i + block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i + 1:i + 1 + block_size]).astype(np.int64)) for i in ix])
        if device_type == 'cuda':
            x = x.pin_memory().to(device, non_blocking=True)
            y = y.pin_memory().to(device, non_blocking=True)
        else:
            x = x.to(device)
            y = y.to(device)
        return x, y

else:
    # fixed synthetic data to isolate model compute
    x = torch.randint(vocab_size, (batch_size, block_size), device=device)
    y = torch.randint(vocab_size, (batch_size, block_size), device=device)
    get_batch = lambda split: (x, y)

def get_prompt(length, bs):
    if real_data:
        if length >= len(train_data) - 1:
            raise ValueError(f"Requested length={length} exceeds dataset capacity.")
        ix = torch.randint(len(train_data) - length - 1, (bs,))
        x = torch.stack([torch.from_numpy((train_data[i:i + length]).astype(np.int64)) for i in ix])
        if device_type == 'cuda':
            x = x.pin_memory().to(device, non_blocking=True)
        else:
            x = x.to(device)
        return x
    return torch.randint(vocab_size, (bs, length), device=device, dtype=torch.long)

if bench_task not in {'train', 'long_context'}:
    raise ValueError(f"Unknown bench_task={bench_task}. Use 'train' or 'long_context'.")

wb = None
if wandb_log:
    try:
        import wandb
    except Exception as e:
        raise RuntimeError(f"wandb_log=True but wandb import failed: {e}")
    wandb_init_kwargs = dict(
        project=wandb_project,
        name=wandb_run_name,
        group=wandb_group,
        config=dict(
            bench_task=bench_task,
            bench_modes=bench_modes,
            dataset=dataset,
            real_data=real_data,
            seed=seed,
            device=device,
            dtype=dtype,
            compile=compile,
            compile_mode=compile_mode,
            compile_disable_cudagraphs=compile_disable_cudagraphs,
            batch_size=batch_size,
            block_size=block_size,
            kv_window=kv_window,
            long_context_lengths=long_context_lengths,
            long_context_batch_size=long_context_batch_size,
            long_context_use_stream=long_context_use_stream,
            experiment_suite=experiment_suite,
        ),
    )
    if wandb_entity:
        wandb_init_kwargs["entity"] = wandb_entity
    wb = wandb.init(**wandb_init_kwargs)

if bench_task == 'train':
    results = []
    profile_total_steps = 15  # wait + warmup + active in the profiler schedule below
    total_train_steps = len(modes) * (profile_total_steps if profile else (burnin_steps + bench_steps))
    bench_bar = make_progress(total=total_train_steps, desc="bench train", leave=True)
    train_csv_cols = [
        "mode", "time_per_iter_ms", "peak_alloc_gb", "peak_reserved_gb",
        "params_total", "tflops_local",
        "seed", "kv_window", "dataset", "timestamp",
    ]
    for mode in modes:
        ctprint(f"\n=== Benchmark mode: {mode} ===")
        model = build_model(mode).to(device)
        raw_model = model
        compute_terms = estimate_model_compute_terms(raw_model)
        fwdbwd_per_iter_local = int(batch_size)
        flops_per_iter_local = float(compute_terms["flops_per_fwdbwd"] * fwdbwd_per_iter_local)
        ctprint(
            "compute estimate: "
            f"params_total={compute_terms['n_params_total']:,}, "
            f"params_non_emb={compute_terms['n_params_non_emb']:,}, "
            f"flops/token={compute_terms['flops_per_token']:.3e}, "
            f"flops/iter={flops_per_iter_local:.3e}"
        )
        optimizer = model.configure_optimizers(
            weight_decay=weight_decay,
            learning_rate=learning_rate,
            betas=(beta1, beta2),
            device_type=device_type,
        )

        if compile:
            ctprint("Compiling model...")
            model = maybe_compile_model(model)

        if profile:
            # useful docs:
            # - tutorial https://pytorch.org/tutorials/intermediate/tensorboard_profiler_tutorial.html
            # - api https://pytorch.org/docs/stable/profiler.html#torch.profiler.profile
            wait, warmup, active = 5, 5, 5
            num_steps = wait + warmup + active
            activities = [torch.profiler.ProfilerActivity.CPU]
            if device_type == 'cuda':
                activities.append(torch.profiler.ProfilerActivity.CUDA)
            trace_dir = os.path.join('./bench_log', mode)
            with torch.profiler.profile(
                activities=activities,
                schedule=torch.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=1),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(trace_dir),
                record_shapes=False,
                profile_memory=False,
                with_stack=False,
                with_flops=True,
                with_modules=False,
            ) as prof:
                X, Y = get_batch('train')
                for k in range(num_steps):
                    with ctx:
                        _, loss = model(X, Y)
                    X, Y = get_batch('train')
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                    bench_bar.update(1)
                    bench_bar.set_postfix(mode=mode, stage="profile", loss=f"{loss.item():.4f}")
                    prof.step()
            ctprint(f"profile traces written to {trace_dir}")
        else:
            sync_if_cuda()
            mode_time_ms = None
            mode_peak_alloc_gb = float('nan')
            mode_peak_reserved_gb = float('nan')
            mode_est_tflops = float('nan')
            for stage, num_steps in enumerate([burnin_steps, bench_steps]):
                reset_peak_if_cuda()
                t0 = time.time()
                X, Y = get_batch('train')
                for k in range(num_steps):
                    with ctx:
                        _, loss = model(X, Y)
                    X, Y = get_batch('train')
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                    bench_bar.update(1)
                    bench_bar.set_postfix(mode=mode, stage="bench", loss=f"{loss.item():.4f}")
                sync_if_cuda()
                dt = time.time() - t0
                if stage == 1:
                    mode_time_ms = dt / num_steps * 1000.0
                    mode_peak_alloc_gb = peak_mem_gb()
                    mode_peak_reserved_gb = peak_reserved_mem_gb()
                    mode_est_tflops = (flops_per_iter_local / max(mode_time_ms * 1e-3, 1e-12)) / 1e12
                    ctprint(f"time per iteration: {mode_time_ms:.4f}ms, TFLOPS(local): {mode_est_tflops:.3f}")
            if mode_time_ms is not None:
                results.append((
                    mode,
                    mode_time_ms,
                    mode_peak_alloc_gb,
                    mode_peak_reserved_gb,
                    compute_terms["n_params_total"],
                    compute_terms["n_params_non_emb"],
                    compute_terms["flops_per_token"],
                    flops_per_iter_local,
                    mode_est_tflops,
                ))
                if wb is not None:
                    log_dict = {
                        "bench/mode": mode,
                        "bench/train_time_per_iter_ms": mode_time_ms,
                        "bench/train_peak_alloc_gb": mode_peak_alloc_gb,
                        "bench/train_peak_reserved_gb": mode_peak_reserved_gb,
                        "compute/params_total": int(compute_terms["n_params_total"]),
                        "compute/params_non_embedding": int(compute_terms["n_params_non_emb"]),
                        "compute/est_flops_per_token": float(compute_terms["flops_per_token"]),
                        "compute/est_flops_per_iter_local": float(flops_per_iter_local),
                        "compute/est_tflops_local": float(mode_est_tflops),
                    }
                    if device_peak_tflops > 0.0:
                        log_dict["compute/est_util_local_pct"] = float(100.0 * mode_est_tflops / device_peak_tflops)
                    wb.log(log_dict)
                csv_append(output_csv, train_csv_cols, {
                    "mode": mode,
                    "time_per_iter_ms": f"{mode_time_ms:.4f}",
                    "peak_alloc_gb": f"{mode_peak_alloc_gb:.4f}",
                    "peak_reserved_gb": f"{mode_peak_reserved_gb:.4f}",
                    "params_total": compute_terms["n_params_total"],
                    "tflops_local": f"{mode_est_tflops:.3f}",
                    "seed": seed,
                    "kv_window": kv_window,
                    "dataset": dataset,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })

        # release memory before next mode
        del model
        del raw_model
        del optimizer
        if device_type == 'cuda':
            torch.cuda.empty_cache()
    bench_bar.close()

    if (not profile) and len(results) > 0:
        cprint("\n=== Benchmark summary ===")
        cprint("mode,time_per_iter_ms,peak_alloc_gb,peak_reserved_gb,params_total,params_non_emb,flops_per_token,flops_per_iter,tflops_local")
        for mode, t_ms, peak_alloc_gb, peak_reserved_gb, p_tot, p_non, flops_tok, flops_iter, tflops_local in results:
            cprint(f"{mode},{t_ms:.4f},{peak_alloc_gb:.2f},{peak_reserved_gb:.2f},{p_tot},{p_non},{flops_tok:.3e},{flops_iter:.3e},{tflops_local:.3f}")
        if wb is not None:
            table = wandb.Table(columns=[
                "mode",
                "time_per_iter_ms",
                "peak_alloc_gb",
                "peak_reserved_gb",
                "params_total",
                "params_non_emb",
                "flops_per_token",
                "flops_per_iter",
                "tflops_local",
            ])
            for mode, t_ms, peak_alloc_gb, peak_reserved_gb, p_tot, p_non, flops_tok, flops_iter, tflops_local in results:
                table.add_data(mode, t_ms, peak_alloc_gb, peak_reserved_gb, p_tot, p_non, flops_tok, flops_iter, tflops_local)
            wb.log({"bench/train_table": table})
            # Train-task summary metrics
            for mode, t_ms, peak_alloc_gb, peak_reserved_gb, p_tot, p_non, flops_tok, flops_iter, tflops_local in results:
                wb.summary[f"bench/train/{mode}/time_per_iter_ms"] = t_ms
                wb.summary[f"bench/train/{mode}/peak_alloc_gb"] = peak_alloc_gb
                wb.summary[f"bench/train/{mode}/peak_reserved_gb"] = peak_reserved_gb
                wb.summary[f"bench/train/{mode}/params_total"] = p_tot
                wb.summary[f"bench/train/{mode}/params_non_embedding"] = p_non
                wb.summary[f"bench/train/{mode}/est_flops_per_token"] = flops_tok
                wb.summary[f"bench/train/{mode}/tflops_local"] = tflops_local

else:  # bench_task == 'long_context'
    lengths = parse_int_csv(long_context_lengths)
    cprint("mode,length,status,peak_alloc_gb_prefill,peak_alloc_gb_decode,prefill_tok_s,decode_tok_s,ttft_ms,decode_ms_per_tok")
    long_rows = []
    long_bar = make_progress(total=len(modes) * len(lengths), desc="bench long", leave=True)
    mode_compute_terms = {}
    lc_csv_cols = [
        "mode", "context_length", "status",
        "peak_alloc_gb", "peak_reserved_gb",
        "peak_alloc_gb_prefill", "peak_alloc_gb_decode",
        "prefill_tok_s", "decode_tok_s",
        "ttft_ms", "decode_ms_per_tok",
        "seed", "kv_window", "dataset", "timestamp",
    ]

    for mode in modes:
        model = build_model(mode).to(device)
        model.eval()
        compute_terms = estimate_model_compute_terms(model)
        mode_compute_terms[mode] = compute_terms
        if wb is not None:
            wb.log({
                "bench/mode": mode,
                "compute/params_total": int(compute_terms["n_params_total"]),
                "compute/params_non_embedding": int(compute_terms["n_params_non_emb"]),
                "compute/est_flops_per_token": float(compute_terms["flops_per_token"]),
                "compute/est_flops_per_fwdbwd": float(compute_terms["flops_per_fwdbwd"]),
            })
        if compile:
            ctprint(f"Compiling model for mode={mode}...")
            model = maybe_compile_model(model)

        for L in lengths:
            status = "ok"
            peak_alloc_gb_prefill = float('nan')
            peak_reserved_gb_prefill = float('nan')
            peak_alloc_gb_decode = float('nan')
            prefill_tok_s = float('nan')
            decode_tok_s = float('nan')
            ttft_ms = float('nan')
            decode_ms_per_tok = float('nan')
            try:
                x = get_prompt(L, long_context_batch_size)
                if device_type == 'cuda':
                    torch.cuda.empty_cache()
                reset_peak_if_cuda()
                sync_if_cuda()
                t0 = time.time()
                state = None
                can_stream = long_context_use_stream and all(hasattr(model, n) for n in ("stream_prefill", "init_stream_state"))
                with torch.no_grad():
                    with ctx:
                        if can_stream:
                            state = model.init_stream_state(x.size(0), x.device, model.transformer.wte.weight.dtype)
                            logits, state = model.stream_prefill(x, state)
                        else:
                            logits, _ = model(x, None)
                sync_if_cuda()
                dt = time.time() - t0
                if dt < 1e-6:
                    ctprint(f"# warn: mode={mode} L={L} prefill timing suspect (dt={dt:.2e}s)")
                    prefill_tok_s = float('nan')
                else:
                    prefill_tok_s = (x.size(0) * x.size(1)) / dt
                    ttft_ms = dt * 1000.0
                peak_alloc_gb_prefill = peak_mem_gb()
                peak_reserved_gb_prefill = peak_reserved_mem_gb()

                # --- decode phase ---
                if bench_decode and can_stream and state is not None:
                    # reset peak stats so decode is measured independently
                    reset_peak_if_cuda()
                    sync_if_cuda()
                    # seed the first decode token from the last prefill logit
                    next_tok = logits[:, -1:, :].argmax(dim=-1)  # [B, 1]
                    t0_dec = time.time()
                    with torch.no_grad():
                        with ctx:
                            for _step in range(decode_steps):
                                logits_dec, state = model.stream_step(next_tok, state)
                                next_tok = logits_dec[:, -1:, :].argmax(dim=-1)
                    sync_if_cuda()
                    dt_dec = time.time() - t0_dec
                    if dt_dec < 1e-6:
                        ctprint(f"# warn: mode={mode} L={L} decode timing suspect (dt_dec={dt_dec:.2e}s)")
                        decode_tok_s = float('nan')
                    else:
                        decode_tok_s = (x.size(0) * decode_steps) / dt_dec
                        decode_ms_per_tok = (dt_dec * 1000.0) / (x.size(0) * decode_steps)
                    peak_alloc_gb_decode = peak_mem_gb()
            except RuntimeError as e:
                msg = str(e).lower()
                if "out of memory" in msg or "cuda out of memory" in msg:
                    status = "oom"
                else:
                    status = "error"
                if long_context_print_errors:
                    ctprint(f"# mode={mode} L={L} err={e}")
                if device_type == 'cuda':
                    peak_alloc_gb_prefill = peak_mem_gb() if np.isnan(peak_alloc_gb_prefill) else peak_alloc_gb_prefill
                    peak_reserved_gb_prefill = peak_reserved_mem_gb() if np.isnan(peak_reserved_gb_prefill) else peak_reserved_gb_prefill
                    torch.cuda.empty_cache()
            except Exception as e:
                status = "error"
                if long_context_print_errors:
                    ctprint(f"# mode={mode} L={L} err={e}")
                if device_type == 'cuda':
                    peak_alloc_gb_prefill = peak_mem_gb() if np.isnan(peak_alloc_gb_prefill) else peak_alloc_gb_prefill
                    peak_reserved_gb_prefill = peak_reserved_mem_gb() if np.isnan(peak_reserved_gb_prefill) else peak_reserved_gb_prefill
                    torch.cuda.empty_cache()

            # combined peak for backward compat
            _p = peak_alloc_gb_prefill if not np.isnan(peak_alloc_gb_prefill) else float('nan')
            _d = peak_alloc_gb_decode if not np.isnan(peak_alloc_gb_decode) else float('nan')
            if np.isnan(_p) and np.isnan(_d):
                peak_gb = float('nan')
            elif np.isnan(_p):
                peak_gb = _d
            elif np.isnan(_d):
                peak_gb = _p
            else:
                peak_gb = max(_p, _d)
            peak_reserved_gb = peak_reserved_gb_prefill  # prefill dominates reservation

            ctprint(f"{mode},{L},{status},{peak_alloc_gb_prefill:.2f},{peak_alloc_gb_decode:.2f},{prefill_tok_s:.1f},{decode_tok_s:.1f},{ttft_ms:.1f},{decode_ms_per_tok:.3f}")
            row = {
                "mode": mode,
                "length": int(L),
                "status": status,
                "peak_gb": float(peak_gb),
                "peak_reserved_gb": float(peak_reserved_gb),
                "peak_alloc_gb_prefill": float(peak_alloc_gb_prefill),
                "peak_alloc_gb_decode": float(peak_alloc_gb_decode),
                "prefill_tok_s": float(prefill_tok_s),
                "decode_tok_s": float(decode_tok_s),
                "ttft_ms": float(ttft_ms),
                "decode_ms_per_tok": float(decode_ms_per_tok),
            }
            long_rows.append(row)
            long_bar.update(1)
            long_bar.set_postfix(mode=mode, ctx=str(L), status=status)
            if wb is not None:
                wb.log(
                    {
                        "bench/mode": mode,
                        "bench/length": int(L),
                        "bench/status": status,
                        "bench/peak_alloc_gb": float(peak_gb),
                        "bench/peak_reserved_gb": float(peak_reserved_gb),
                        "bench/peak_alloc_gb_prefill": float(peak_alloc_gb_prefill),
                        "bench/peak_alloc_gb_decode": float(peak_alloc_gb_decode),
                        "bench/prefill_tok_s": float(prefill_tok_s),
                        "bench/decode_tok_s": float(decode_tok_s),
                        "bench/ttft_ms": float(ttft_ms),
                        "bench/decode_ms_per_tok": float(decode_ms_per_tok),
                    }
                )
            csv_append(output_csv, lc_csv_cols, {
                "mode": mode,
                "context_length": int(L),
                "status": status,
                "peak_alloc_gb": f"{peak_gb:.4f}",
                "peak_reserved_gb": f"{peak_reserved_gb:.4f}",
                "peak_alloc_gb_prefill": f"{peak_alloc_gb_prefill:.4f}",
                "peak_alloc_gb_decode": f"{peak_alloc_gb_decode:.4f}",
                "prefill_tok_s": f"{prefill_tok_s:.1f}",
                "decode_tok_s": f"{decode_tok_s:.1f}",
                "ttft_ms": f"{ttft_ms:.1f}",
                "decode_ms_per_tok": f"{decode_ms_per_tok:.3f}",
                "seed": seed,
                "kv_window": kv_window,
                "dataset": dataset,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
        del model
        if device_type == 'cuda':
            torch.cuda.empty_cache()

    long_bar.close()

    if wb is not None and len(long_rows) > 0:
        table = wandb.Table(columns=["mode", "length", "status", "peak_alloc_gb", "peak_reserved_gb", "peak_alloc_gb_prefill", "peak_alloc_gb_decode", "prefill_tok_s", "decode_tok_s"])
        for row in long_rows:
            table.add_data(
                row["mode"],
                row["length"],
                row["status"],
                row["peak_gb"],
                row["peak_reserved_gb"],
                row["peak_alloc_gb_prefill"],
                row["peak_alloc_gb_decode"],
                row["prefill_tok_s"],
                row["decode_tok_s"],
            )
        wb.log({"bench/long_context_table": table})

        for mode in modes:
            rows_m = [r for r in long_rows if r["mode"] == mode]
            if not rows_m:
                continue
            ok_lengths = [r["length"] for r in rows_m if r["status"] == "ok"]
            oom_lengths = [r["length"] for r in rows_m if r["status"] == "oom"]
            max_ok_len = max(ok_lengths) if ok_lengths else 0
            first_oom_len = min(oom_lengths) if oom_lengths else 0
            max_peak_alloc = max((r["peak_gb"] for r in rows_m if np.isfinite(r["peak_gb"])), default=float('nan'))
            max_peak_reserved = max((r["peak_reserved_gb"] for r in rows_m if np.isfinite(r["peak_reserved_gb"])), default=float('nan'))
            wb.summary[f"bench/{mode}/max_ok_length"] = int(max_ok_len)
            wb.summary[f"bench/{mode}/first_oom_length"] = int(first_oom_len)
            wb.summary[f"bench/{mode}/max_peak_alloc_gb"] = float(max_peak_alloc)
            wb.summary[f"bench/{mode}/max_peak_reserved_gb"] = float(max_peak_reserved)
            c = mode_compute_terms.get(mode, None)
            if c is not None:
                wb.summary[f"bench/{mode}/params_total"] = int(c["n_params_total"])
                wb.summary[f"bench/{mode}/params_non_embedding"] = int(c["n_params_non_emb"])
                wb.summary[f"bench/{mode}/est_flops_per_token"] = float(c["flops_per_token"])

if wb is not None:
    wb.finish()
