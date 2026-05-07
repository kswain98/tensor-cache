"""
Sample from a trained model
"""
import os
import pickle
import time
from contextlib import nullcontext
import torch
import tiktoken
import torch.nn.functional as F
from model import GPTConfig, GPT
from utils import apply_overrides


# -----------------------------------------------------------------------------
init_from = 'resume' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir = 'out' # ignored if init_from is not 'resume'
start = "\n" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_samples = 10 # number of samples to draw
max_new_tokens = 500 # number of tokens generated in each sample
temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1337
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster
kv_mode = 'checkpoint' # {'checkpoint', 'full_kv', 'window_kv', 'streaming_llm', 'infini', 'tc'}
kv_window = 512 # used when kv_mode is 'window_kv' or 'tc'
tc_write_on_evict = True
tc_write_on_insert = False
wandb_log = False
wandb_project = 'owt'
wandb_run_name = 'sample'
wandb_entity = '' # optionally set your wandb entity/team
wandb_group = 'inference'
wandb_log_text = True
wandb_max_text_chars = 4000
experiment_suite = 'long-context'
wandb_reset_peak_each_sample = True
apply_overrides(globals()) # overrides from command line or config file
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# model
if init_from == 'resume':
    # init from a model saved in a specific directory
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    gptconf = GPTConfig(**checkpoint_model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
elif init_from.startswith('gpt2'):
    # init from a given GPT-2 model
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))

if kv_mode == 'full_kv':
    model.cfg.use_kv_cache = True
    model.cfg.kv_window = 0
    model.cfg.tc_enabled = False
elif kv_mode == 'window_kv':
    if kv_window <= 0:
        raise ValueError("kv_mode='window_kv' requires kv_window > 0.")
    model.cfg.use_kv_cache = True
    model.cfg.kv_window = kv_window
    model.cfg.tc_enabled = False
elif kv_mode == 'streaming_llm':
    if kv_window <= 0:
        raise ValueError("kv_mode='streaming_llm' requires kv_window > 0.")
    model.cfg.use_kv_cache = True
    model.cfg.kv_window = kv_window
    model.cfg.kv_num_sinks = max(model.cfg.kv_num_sinks, 4)
    model.cfg.tc_enabled = False
elif kv_mode == 'infini':
    if kv_window <= 0:
        raise ValueError("kv_mode='infini' requires kv_window > 0.")
    model.cfg.use_kv_cache = True
    model.cfg.kv_window = kv_window
    model.cfg.tc_enabled = False
    model.cfg.infini_enabled = True
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
        raise ValueError("kv_mode='tc' requires at least one TC write path: tc_write_on_evict or tc_write_on_insert.")
elif kv_mode != 'checkpoint':
    raise ValueError(f"Unknown kv_mode: {kv_mode}")

model.eval()
model.to(device)
if compile:
    model = torch.compile(model) # requires PyTorch 2.0 (optional)

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
            init_from=init_from,
            out_dir=out_dir,
            kv_mode=kv_mode,
            kv_window=kv_window,
            device=device,
            dtype=dtype,
            compile=compile,
            num_samples=num_samples,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            experiment_suite=experiment_suite,
            seed=seed,
        ),
    )
    if wandb_entity:
        wandb_init_kwargs["entity"] = wandb_entity
    wb = wandb.init(**wandb_init_kwargs)

# look for the meta pickle in case it is available in the dataset folder
load_meta = False
if init_from == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint['config']: # older checkpoints might not have these...
    meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
    load_meta = os.path.exists(meta_path)
if load_meta:
    print(f"Loading meta from {meta_path}...")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
else:
    # ok let's assume gpt-2 encodings by default
    print("No meta.pkl found, assuming GPT-2 encodings...")
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)

# encode the beginning of the prompt
if start.startswith('FILE:'):
    with open(start[5:], 'r', encoding='utf-8') as f:
        start = f.read()
start_ids = encode(start)
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

def _cuda_sync_if_needed():
    if device_type == 'cuda':
        torch.cuda.synchronize()

@torch.no_grad()
def generate_with_timing(x_in: torch.Tensor):
    use_stream = all(hasattr(model, n) for n in ('init_stream_state', 'stream_prefill', 'stream_step'))

    if not use_stream:
        _cuda_sync_if_needed()
        t0 = time.time()
        y_out = model.generate(x_in, max_new_tokens, temperature=temperature, top_k=top_k)
        _cuda_sync_if_needed()
        t1 = time.time()
        total_s = t1 - t0
        decode_tok_s = float(max_new_tokens) / max(total_s, 1e-12)
        return y_out, {
            "path": "generate",
            "prefill_s": float("nan"),
            "decode_s": total_s,
            "total_s": total_s,
            "prefill_tok_s": float("nan"),
            "decode_tok_s": decode_tok_s,
        }

    B, Tprompt = x_in.shape
    state = model.init_stream_state(B, x_in.device, model.transformer.wte.weight.dtype)

    _cuda_sync_if_needed()
    t0 = time.time()
    logits, state = model.stream_prefill(x_in, state)
    _cuda_sync_if_needed()
    t1 = time.time()

    out = x_in
    for _ in range(max_new_tokens):
        next_logits = logits[:, -1, :] / max(1e-6, float(temperature))
        if top_k is not None:
            v, _ = torch.topk(next_logits, min(int(top_k), next_logits.size(-1)))
            next_logits[next_logits < v[:, [-1]]] = -float("inf")
        probs = F.softmax(next_logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        out = torch.cat([out, next_id], dim=1)
        logits, state = model.stream_step(next_id, state)

    _cuda_sync_if_needed()
    t2 = time.time()
    prefill_s = t1 - t0
    decode_s = t2 - t1
    total_s = t2 - t0
    return out, {
        "path": "stream",
        "prefill_s": prefill_s,
        "decode_s": decode_s,
        "total_s": total_s,
        "prefill_tok_s": float(Tprompt) / max(prefill_s, 1e-12),
        "decode_tok_s": float(max_new_tokens) / max(decode_s, 1e-12),
    }

# run generation
sample_rows = []
run_peak_alloc_gb = 0.0
run_peak_reserved_gb = 0.0
with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            if wb is not None and device_type == 'cuda' and wandb_reset_peak_each_sample:
                torch.cuda.reset_peak_memory_stats()
            y, timing = generate_with_timing(x)
            text = decode(y[0].tolist())
            print(text)
            print('---------------')

            generated_tokens = int(y.size(1) - x.size(1))
            total_tok_s = float(x.size(1) + generated_tokens) / max(float(timing["total_s"]), 1e-12)
            peak_alloc_gb = torch.cuda.max_memory_allocated() / (1024**3) if device_type == 'cuda' else float('nan')
            peak_reserved_gb = torch.cuda.max_memory_reserved() / (1024**3) if device_type == 'cuda' else float('nan')
            if device_type == 'cuda':
                run_peak_alloc_gb = max(run_peak_alloc_gb, peak_alloc_gb)
                run_peak_reserved_gb = max(run_peak_reserved_gb, peak_reserved_gb)
            sample_row = {
                "sample_idx": k,
                "path": timing["path"],
                "prompt_tokens": int(x.size(1)),
                "generated_tokens": generated_tokens,
                "prefill_s": float(timing["prefill_s"]),
                "decode_s": float(timing["decode_s"]),
                "total_s": float(timing["total_s"]),
                "prefill_tok_s": float(timing["prefill_tok_s"]),
                "decode_tok_s": float(timing["decode_tok_s"]),
                "total_tok_s": total_tok_s,
                "peak_alloc_gb": float(peak_alloc_gb),
                "peak_reserved_gb": float(peak_reserved_gb),
                "text": text,
            }
            sample_rows.append(sample_row)

            if wb is not None:
                log_dict = {
                    "sample/index": k,
                    "sample/path": timing["path"],
                    "sample/prompt_tokens": int(x.size(1)),
                    "sample/generated_tokens": generated_tokens,
                    "inference/prefill_s": float(timing["prefill_s"]),
                    "inference/decode_s": float(timing["decode_s"]),
                    "inference/total_s": float(timing["total_s"]),
                    "inference/prefill_tok_s": float(timing["prefill_tok_s"]),
                    "inference/decode_tok_s": float(timing["decode_tok_s"]),
                    "inference/total_tok_s": total_tok_s,
                }
                if device_type == 'cuda':
                    log_dict.update({
                        "system/gpu_mem_alloc_gb": torch.cuda.memory_allocated() / (1024**3),
                        "system/gpu_mem_reserved_gb": torch.cuda.memory_reserved() / (1024**3),
                        "system/gpu_mem_peak_sample_alloc_gb": peak_alloc_gb,
                        "system/gpu_mem_peak_sample_reserved_gb": peak_reserved_gb,
                        "system/gpu_mem_peak_run_alloc_gb": run_peak_alloc_gb,
                        "system/gpu_mem_peak_run_reserved_gb": run_peak_reserved_gb,
                    })
                wb.log(log_dict, step=k)
                if wandb_log_text:
                    wb.log({
                        "sample/text": text[:int(wandb_max_text_chars)],
                    }, step=k)

if wb is not None and len(sample_rows) > 0:
    import wandb
    cols = [
        "sample_idx", "path", "prompt_tokens", "generated_tokens",
        "prefill_s", "decode_s", "total_s",
        "prefill_tok_s", "decode_tok_s", "total_tok_s",
        "peak_alloc_gb", "peak_reserved_gb", "text",
    ]
    table = wandb.Table(columns=cols)
    for row in sample_rows:
        table.add_data(*[row[c] if c != "text" else row[c][:int(wandb_max_text_chars)] for c in cols])
    wb.log({"inference/samples_table": table})

    avg_decode_tok_s = sum(r["decode_tok_s"] for r in sample_rows) / len(sample_rows)
    avg_total_tok_s = sum(r["total_tok_s"] for r in sample_rows) / len(sample_rows)
    wb.summary["inference/avg_decode_tok_s"] = avg_decode_tok_s
    wb.summary["inference/avg_total_tok_s"] = avg_total_tok_s
    wb.summary["inference/num_samples"] = len(sample_rows)
    if device_type == 'cuda':
        wb.summary["system/gpu_mem_peak_run_alloc_gb"] = run_peak_alloc_gb
        wb.summary["system/gpu_mem_peak_run_reserved_gb"] = run_peak_reserved_gb
    wb.finish()
