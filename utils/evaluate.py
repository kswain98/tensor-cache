"""
Distance-stratified long-context NLL/PPL evaluation.

Evaluates a contiguous token stream and reports:
- nll_all / ppl_all
- nll_far / ppl_far      (positions >= far_start)
- nll_vfar / ppl_vfar    (positions >= very_far_start)
"""

import math
import os
import sys
import pickle
import time
from contextlib import nullcontext
from pathlib import Path

# Make project root importable so `tensor_cache.*`, `utils.*`, `baselines.*` resolve
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import tiktoken
import torch
import torch.nn.functional as F

from tensor_cache.model import GPTConfig, GPT
from utils import make_progress, progress_enabled, cprint, apply_overrides

# -----------------------------------------------------------------------------
# model/checkpoint
init_from = 'resume'  # {'resume', 'gpt2*'}
out_dir = 'out'
kv_mode = 'checkpoint'  # {'checkpoint', 'full_kv', 'window_kv', 'tc', 'streaming_llm', 'infini'}
kv_window = 512

tc_write_on_evict = True
tc_write_on_insert = False

# data/eval
# If dataset is empty and init_from='resume', dataset is read from ckpt config.
dataset = ''
split = 'val'  # {'train', 'val'}
seed = 1337
eval_tokens = 32768
eval_start = -1  # -1 => random valid start
far_start = 1024       # 2x default kv_window (512); positions with eviction history
very_far_start = 4096  # 8x default kv_window; positions with deep eviction history

# runtime
device = 'cuda'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = False
compile_mode = 'default'
compile_disable_cudagraphs = False
log_interval = 2000

# metadata/logging
experiment_suite = 'long-context'
wandb_log = False
wandb_project = 'owt'
wandb_run_name = 'quality-eval'
wandb_entity = ''
wandb_group = 'quality'
output_csv = ''  # optional csv path
output_csv_long = ''  # optional long-format csv path

# configurable distance bins
# distance_bins: comma-separated values, e.g. "1,2,4,8" (window_mult) or "512,1024,2048" (absolute)
# empty string = legacy 3-bin behavior only
distance_bins = ''
distance_bins_mode = 'window_mult'  # 'window_mult' or 'absolute'

apply_overrides(globals())
# -----------------------------------------------------------------------------


def maybe_compile_model(model, device_type):
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


def safe_avg(total, count):
    if count <= 0:
        return float('nan')
    return float(total) / float(count)


def ppl(nll):
    if not np.isfinite(nll):
        return float('nan')
    try:
        return float(math.exp(float(nll)))
    except OverflowError:
        return float('inf')


def nll_to_bpc(nll, chars_per_token):
    """Convert NLL (nats/token) to BPC (bits/character).

    BPC = NLL / ln(2) / chars_per_token
    """
    if not np.isfinite(nll) or chars_per_token <= 0:
        return float('nan')
    return float(nll) / math.log(2) / chars_per_token


def append_csv(path, row, columns):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    write_header = (not os.path.exists(path)) or (os.path.getsize(path) == 0)
    with open(path, 'a', encoding='utf-8') as f:
        if write_header:
            f.write(','.join(columns) + '\n')
        vals = []
        for c in columns:
            v = row[c]
            if isinstance(v, float):
                vals.append(f"{v:.6f}")
            else:
                vals.append(str(v))
        f.write(','.join(vals) + '\n')


def parse_distance_bins(bins_str, mode, window):
    """Parse distance_bins config into a sorted list of bin boundaries.

    Returns list of ints, e.g. [512, 1024, 2048, 4096].
    Empty list when bins_str is empty (legacy mode).
    """
    if not bins_str or not bins_str.strip():
        return []
    parts = [s.strip() for s in bins_str.split(',') if s.strip()]
    boundaries = []
    for p in parts:
        v = float(p)
        if mode == 'window_mult':
            boundaries.append(int(v * window))
        else:  # absolute
            boundaries.append(int(v))
    boundaries = sorted(set(boundaries))
    return boundaries


def main():
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device_type = 'cuda' if 'cuda' in device else 'cpu'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    checkpoint = None

    if init_from == 'resume':
        ckpt_path = os.path.join(out_dir, 'ckpt.pt')
        checkpoint = torch.load(ckpt_path, map_location=device)
        ckpt_args = checkpoint['model_args']
        gptconf = GPTConfig(**ckpt_args)
        model = GPT(gptconf)
        state_dict = checkpoint['model']
        unwanted_prefix = '_orig_mod.'
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
    elif init_from.startswith('gpt2'):
        model = GPT.from_pretrained(init_from, dict(dropout=0.0))
    else:
        raise ValueError(f"Unsupported init_from={init_from}")

    if kv_mode == 'full_kv':
        model.cfg.use_kv_cache = True
        model.cfg.kv_window = 0
        model.cfg.tc_enabled = False
        # NTK-aware RoPE scaling for context beyond training block_size
        if eval_tokens > model.cfg.block_size and hasattr(model, 'set_rope_scaling'):
            factor = eval_tokens / model.cfg.block_size
            model.set_rope_scaling(factor)
            cprint(f"Applied RoPE scaling factor={factor:.1f} for full_kv "
                   f"(eval_tokens={eval_tokens} > block_size={model.cfg.block_size})")
    elif kv_mode == 'window_kv':
        if kv_window <= 0:
            raise ValueError("kv_mode='window_kv' requires kv_window > 0.")
        model.cfg.use_kv_cache = True
        model.cfg.kv_window = kv_window
        model.cfg.tc_enabled = False
    elif kv_mode == 'streaming_llm':
        if kv_window <= 0:
            raise ValueError("kv_mode='streaming_llm' requires kv_window > 0.")
        model.cfg.apply_kv_mode('streaming_llm')
        model.cfg.kv_window = kv_window
    elif kv_mode == 'tc':
        if kv_window <= 0:
            raise ValueError("kv_mode='tc' requires kv_window > 0.")
        if not any(hasattr(block.attn, 'tc') for block in model.transformer.h):
            raise ValueError("kv_mode='tc' requires a checkpoint/model built with tc_enabled=True.")
        model.cfg.use_kv_cache = True
        model.cfg.kv_window = kv_window
        model.cfg.tc_enabled = True
        model.cfg.tc_write_on_evict = bool(tc_write_on_evict)
        model.cfg.tc_write_on_insert = bool(tc_write_on_insert)
        if (not model.cfg.tc_write_on_evict) and (not model.cfg.tc_write_on_insert):
            raise ValueError("kv_mode='tc' requires at least one TC write path.")
    elif kv_mode == 'infini':
        if kv_window <= 0:
            raise ValueError("kv_mode='infini' requires kv_window > 0.")
        if not any(hasattr(block.attn, 'infini') for block in model.transformer.h):
            raise ValueError("kv_mode='infini' requires a checkpoint/model built with infini_enabled=True.")
        model.cfg.apply_kv_mode('infini')
        model.cfg.kv_window = kv_window
    elif kv_mode != 'checkpoint':
        raise ValueError(f"Unknown kv_mode: {kv_mode}")

    model.eval()
    model.to(device)
    model = maybe_compile_model(model, device_type)

    # dataset resolution
    if not dataset:
        if checkpoint is not None and 'config' in checkpoint and 'dataset' in checkpoint['config']:
            dataset_name = checkpoint['config']['dataset']
        else:
            raise ValueError("dataset not set and checkpoint does not include config.dataset")
    else:
        dataset_name = dataset

    data_dir = os.path.join('data', dataset_name)
    split_file = 'train.bin' if split == 'train' else 'val.bin'
    data_path = os.path.join(data_dir, split_file)
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Missing dataset split file: {data_path}")

    data = np.memmap(data_path, dtype=np.uint16, mode='r')
    max_start = int(len(data) - int(eval_tokens) - 1)
    if max_start < 0:
        raise ValueError(
            f"eval_tokens={eval_tokens} too large for split size {len(data)}; need at least eval_tokens+1 tokens"
        )
    if int(eval_start) < 0:
        start = int(np.random.randint(0, max_start + 1))
    else:
        start = int(eval_start)
        if start < 0 or start > max_start:
            raise ValueError(f"eval_start={start} out of range [0, {max_start}]")

    stream_tokens = np.asarray(data[start:start + int(eval_tokens) + 1], dtype=np.int64)

    # Compute chars-per-token ratio for BPC. Optional — tiktoken hits the network
    # on first use to fetch the GPT-2 BPE vocab, so a DNS or VPN hiccup here
    # would otherwise abort the entire eval. Fall back to NaN BPC if it fails.
    try:
        enc = tiktoken.get_encoding("gpt2")
        decoded_text = enc.decode(stream_tokens.tolist())
        chars_per_token = len(decoded_text) / len(stream_tokens) if len(stream_tokens) > 0 else 1.0
        cprint(f"[info] chars_per_token={chars_per_token:.3f} (total_chars={len(decoded_text)}, total_tokens={len(stream_tokens)})")
    except Exception as e:
        cprint(f"[warn] tiktoken unavailable ({type(e).__name__}); skipping BPC, NLL/PPL still computed")
        chars_per_token = float('nan')

    wb = None
    if wandb_log:
        try:
            import wandb
        except Exception as e:
            raise RuntimeError(f"wandb_log=True but wandb import failed: {e}")
        init_kwargs = dict(
            project=wandb_project,
            name=wandb_run_name,
            group=wandb_group,
            config=dict(
                init_from=init_from,
                out_dir=out_dir,
                dataset=dataset_name,
                split=split,
                seed=seed,
                kv_mode=kv_mode,
                kv_window=kv_window,
                tc_write_on_evict=tc_write_on_evict,
                tc_write_on_insert=tc_write_on_insert,
                eval_tokens=eval_tokens,
                eval_start=start,
                far_start=far_start,
                very_far_start=very_far_start,
                device=device,
                dtype=dtype,
                compile=compile,
                experiment_suite=experiment_suite,
                distance_bins=distance_bins,
                distance_bins_mode=distance_bins_mode,
            ),
        )
        if wandb_entity:
            init_kwargs['entity'] = wandb_entity
        wb = wandb.init(**init_kwargs)

    nll_all_sum = 0.0
    nll_all_count = 0
    nll_far_sum = 0.0
    nll_far_count = 0
    nll_vfar_sum = 0.0
    nll_vfar_count = 0

    # configurable bins
    bin_boundaries = parse_distance_bins(distance_bins, distance_bins_mode, kv_window)
    use_custom_bins = len(bin_boundaries) > 0
    # bin_edges: [0, b1, b2, ..., inf] => N+1 edges => N bins
    # bin i covers positions in [bin_edges[i], bin_edges[i+1])
    if use_custom_bins:
        bin_edges = [0] + bin_boundaries + [float('inf')]
        n_bins = len(bin_edges) - 1
        bin_nll_sum = [0.0] * n_bins
        bin_nll_count = [0] * n_bins
        cprint(f"[info] custom distance bins: edges={bin_edges}")
    else:
        bin_edges = []
        n_bins = 0
        bin_nll_sum = []
        bin_nll_count = []

    cprint(f"[info] dataset={dataset_name} split={split} start={start} tokens={eval_tokens}")

    if device_type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    t0 = time.time()
    with torch.no_grad():
        state = model.init_stream_state(1, device, model.transformer.wte.weight.dtype)
        token_bar = make_progress(total=int(eval_tokens), desc=f"{kv_mode} eval", leave=True)
        for t in range(int(eval_tokens)):
            idx_t = torch.tensor([[int(stream_tokens[t])]], dtype=torch.long, device=device)
            target_id = int(stream_tokens[t + 1])
            with ctx:
                logits, state = model.stream_step(idx_t, state)
                logp = F.log_softmax(logits[0, 0, :], dim=-1)
                nll = float((-logp[target_id]).item())

            pos = t + 1
            nll_all_sum += nll
            nll_all_count += 1
            if pos >= int(far_start):
                nll_far_sum += nll
                nll_far_count += 1
            if pos >= int(very_far_start):
                nll_vfar_sum += nll
                nll_vfar_count += 1
            if use_custom_bins:
                for bi in range(n_bins):
                    if bin_edges[bi] <= pos < bin_edges[bi + 1]:
                        bin_nll_sum[bi] += nll
                        bin_nll_count[bi] += 1
                        break
            token_bar.update(1)
            if progress_enabled() and (t + 1) % int(log_interval) == 0:
                token_bar.set_postfix(nll=f"{safe_avg(nll_all_sum, nll_all_count):.4f}")
            elif (not progress_enabled()) and (t + 1) % int(log_interval) == 0:
                cprint(f"[eval] {t+1}/{eval_tokens} tokens")
        token_bar.close()

    if device_type == 'cuda':
        torch.cuda.synchronize()
    dt = time.time() - t0
    if dt < 1e-6:
        cprint(f"[warn] eval timing suspect (dt={dt:.2e}s), setting tok_s=nan")
        dt = float('nan')

    nll_all = safe_avg(nll_all_sum, nll_all_count)
    nll_far = safe_avg(nll_far_sum, nll_far_count)
    nll_vfar = safe_avg(nll_vfar_sum, nll_vfar_count)

    # compute per-bin NLL/PPL
    bin_nll_vals = []
    if use_custom_bins:
        for bi in range(n_bins):
            bin_nll_vals.append(safe_avg(bin_nll_sum[bi], bin_nll_count[bi]))

    row = {
        'mode': kv_mode,
        'dataset': dataset_name,
        'split': split,
        'eval_tokens': int(eval_tokens),
        'eval_start': int(start),
        'far_start': int(far_start),
        'very_far_start': int(very_far_start),
        'nll_all': float(nll_all),
        'ppl_all': float(ppl(nll_all)),
        'bpc_all': nll_to_bpc(nll_all, chars_per_token),
        'nll_far': float(nll_far),
        'ppl_far': float(ppl(nll_far)),
        'bpc_far': nll_to_bpc(nll_far, chars_per_token),
        'nll_vfar': float(nll_vfar),
        'ppl_vfar': float(ppl(nll_vfar)),
        'bpc_vfar': nll_to_bpc(nll_vfar, chars_per_token),
        'chars_per_token': chars_per_token,
        'tok_s': float(eval_tokens / dt),
        'peak_alloc_gb': float(torch.cuda.max_memory_allocated() / (1024 ** 3)) if device_type == 'cuda' else float('nan'),
        'peak_reserved_gb': float(torch.cuda.max_memory_reserved() / (1024 ** 3)) if device_type == 'cuda' else float('nan'),
    }

    # add custom bin columns to row
    if use_custom_bins:
        for bi in range(n_bins):
            edge_val = bin_edges[bi] if bin_edges[bi] != float('inf') else -1
            row[f'bin_boundary_{bi}'] = int(edge_val)
            row[f'nll_bin_{bi}'] = float(bin_nll_vals[bi])
            row[f'ppl_bin_{bi}'] = float(ppl(bin_nll_vals[bi]))
            row[f'bpc_bin_{bi}'] = nll_to_bpc(bin_nll_vals[bi], chars_per_token)
            row[f'count_bin_{bi}'] = int(bin_nll_count[bi])

    cprint('mode,dataset,split,eval_tokens,far_start,very_far_start,nll_all,ppl_all,bpc_all,nll_far,ppl_far,bpc_far,nll_vfar,ppl_vfar,bpc_vfar,tok_s,peak_alloc_gb,peak_reserved_gb')
    cprint(
        f"{row['mode']},{row['dataset']},{row['split']},{row['eval_tokens']},{row['far_start']},{row['very_far_start']},"
        f"{row['nll_all']:.6f},{row['ppl_all']:.4f},{row['bpc_all']:.4f},{row['nll_far']:.6f},{row['ppl_far']:.4f},{row['bpc_far']:.4f},"
        f"{row['nll_vfar']:.6f},{row['ppl_vfar']:.4f},{row['bpc_vfar']:.4f},{row['tok_s']:.2f},{row['peak_alloc_gb']:.2f},{row['peak_reserved_gb']:.2f}"
    )
    if use_custom_bins:
        for bi in range(n_bins):
            edge_end = bin_edges[bi + 1] if bin_edges[bi + 1] != float('inf') else 'inf'
            cprint(f"  bin_{bi}: [{bin_edges[bi]}, {edge_end})  nll={bin_nll_vals[bi]:.6f}  ppl={ppl(bin_nll_vals[bi]):.4f}  bpc={nll_to_bpc(bin_nll_vals[bi], chars_per_token):.4f}  count={bin_nll_count[bi]}")

    if output_csv:
        cols = [
            'mode', 'dataset', 'split', 'eval_tokens', 'eval_start', 'far_start', 'very_far_start',
            'nll_all', 'ppl_all', 'bpc_all', 'nll_far', 'ppl_far', 'bpc_far', 'nll_vfar', 'ppl_vfar', 'bpc_vfar',
            'chars_per_token', 'tok_s', 'peak_alloc_gb', 'peak_reserved_gb',
        ]
        if use_custom_bins:
            for bi in range(n_bins):
                cols.extend([f'bin_boundary_{bi}', f'nll_bin_{bi}', f'ppl_bin_{bi}', f'bpc_bin_{bi}', f'count_bin_{bi}'])
        append_csv(output_csv, row, cols)
        cprint(f"[ok] appended results to {output_csv}")

    # long-format CSV: one row per bin
    if output_csv_long and use_custom_bins:
        long_cols = ['mode', 'dataset', 'split', 'position_bin_start', 'position_bin_end', 'nll', 'ppl', 'bpc', 'count']
        for bi in range(n_bins):
            edge_start = int(bin_edges[bi]) if bin_edges[bi] != float('inf') else -1
            edge_end = int(bin_edges[bi + 1]) if bin_edges[bi + 1] != float('inf') else -1
            long_row = {
                'mode': kv_mode,
                'dataset': dataset_name,
                'split': split,
                'position_bin_start': edge_start,
                'position_bin_end': edge_end,
                'nll': float(bin_nll_vals[bi]),
                'ppl': float(ppl(bin_nll_vals[bi])),
                'bpc': nll_to_bpc(bin_nll_vals[bi], chars_per_token),
                'count': int(bin_nll_count[bi]),
            }
            append_csv(output_csv_long, long_row, long_cols)
        cprint(f"[ok] appended {n_bins} bin rows to {output_csv_long}")

    if wb is not None:
        wb.log({f"quality/{k}": v for k, v in row.items() if isinstance(v, (int, float))})
        wb.summary.update({f"quality/{k}": v for k, v in row.items()})
        wb.finish()


if __name__ == '__main__':
    main()
