"""Verify the T4 optimization changes and optionally benchmark the model."""
import argparse
import time
import torch

from nanochat.common import get_base_dir
from nanochat.gpt import GPT, GPTConfig
from nanochat.tokenizer import get_tokenizer


parser = argparse.ArgumentParser()
parser.add_argument("--depth", type=int, default=8)
parser.add_argument("--seq-len", type=int, default=512)
parser.add_argument("--batch-size", type=int, default=4)
parser.add_argument("--benchmark-steps", type=int, default=10)
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = get_tokenizer()
vocab_size = tokenizer.get_vocab_size()
model_dim = args.depth * 64
num_heads = model_dim // 128
config = GPTConfig(
    sequence_len=args.seq_len,
    vocab_size=vocab_size,
    n_layer=args.depth,
    n_head=num_heads,
    n_kv_head=num_heads,
    n_embd=model_dim,
    window_pattern="L",
)

with torch.device("meta"):
    model = GPT(config)

attn = model.transformer.h[0].attn
assert hasattr(attn, "c_qkv"), "Fused QKV is NOT active: c_qkv is missing"
assert not any(hasattr(attn, name) for name in ("c_q", "c_k", "c_v")), "Unfused Q/K/V still present"

print("T4 optimization verification")
print(f"base dir: {get_base_dir()}")
print(f"device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"vocab size: {vocab_size}")
print(f"layers: {config.n_layer}")
print(f"hidden size: {config.n_embd}")
print(f"sequence length: {config.sequence_len}")
print("fused QKV: YES")
print(f"attention pattern: {config.window_pattern} (full attention)")

if device == "cuda" and args.benchmark_steps > 0:
    model = model.to_empty(device="cuda")
    model.init_weights()
    model.train()
    dtype = torch.float16
    model = model.to(dtype=dtype)
    x = torch.randint(0, vocab_size, (args.batch_size, args.seq_len), device="cuda")
    y = torch.roll(x, shifts=-1, dims=1)
    for _ in range(3):
        loss = model(x, y)
        loss.backward()
        model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(args.benchmark_steps):
        loss = model(x, y)
        loss.backward()
        model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    tokens = args.benchmark_steps * args.batch_size * args.seq_len
    print(f"benchmark: {tokens / elapsed:,.0f} tokens/sec")
