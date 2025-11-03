
import torch.nn.functional as F
from contextlib import suppress
import torch, time
from torch.nn.attention import sdpa_kernel, SDPBackend



print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("CUDA version:", torch.version.cuda)
    print("cuDNN version:", torch.backends.cudnn.version())
    print("Number of GPUs:", torch.cuda.device_count())
    print("GPU:", torch.cuda.get_device_name(0))
    print("Capability:", torch.cuda.get_device_capability(0))
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"  Memory Allocated: {torch.cuda.memory_allocated(i) / (1024 ** 3):.2f} GB")
        print(f"  Memory Cached: {torch.cuda.memory_reserved(i) / (1024 ** 3):.2f} GB")
else:
    print("No CUDA-capable device detected.")
# 1) 檢查 SDPA 旗標（是否被啟用；非可用性）
for name in ("flash_sdp_enabled", "mem_efficient_sdp_enabled", "math_sdp_enabled", "cudnn_sdp_enabled"):
    fn = getattr(torch.backends.cuda, name, None)
    if callable(fn):
        print(f"{name}: {fn()}")

# 2) 嘗試用 can_use_flash_attention（較新版本提供）
can = getattr(torch.backends.cuda, "can_use_flash_attention", None)
if callable(can) and torch.cuda.is_available():
    q = torch.randn(1, 8, 64, 64, device="cuda", dtype=torch.float16)
    k = torch.randn_like(q); v = torch.randn_like(q)
    try:
        from torch.backends.cuda import SDPAParams
        ok = can(SDPAParams(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False), debug=True)
        print("can_use_flash_attention:", ok)
    except Exception as e:
        print("can_use_flash_attention: not available in this build ->", e)
else:
    print("can_use_flash_attention: n/a")

# 3) 實際強制要求 FLASH 後端（若可用會成功，否則丟錯）
try:
    from torch.nn.attention import sdpa_kernel as nn_sdpa_kernel, SDPBackend
    q = torch.randn(1, 8, 64, 64, device="cuda", dtype=torch.float16)
    k = torch.randn_like(q); v = torch.randn_like(q)
    with nn_sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        _ = F.scaled_dot_product_attention(q, k, v, is_causal=False)
    print("FLASH backend executed OK")
except Exception as e:
    print("FLASH backend failed:", e)

# 4) 預設路徑跑一次（PyTorch 會自動選可用的最佳後端）
if torch.cuda.is_available():
    q = torch.randn(1, 8, 64, 64, device="cuda", dtype=torch.float16)
    k = torch.randn_like(q); v = torch.randn_like(q)
    with suppress(Exception):
        _ = F.scaled_dot_product_attention(q, k, v, is_causal=False)
    print("Ran SDPA once; PyTorch picked an available backend.")





def bench(B=1,H=16,S=512,D=64, iters=50):
    q = torch.randn(B,H,S,D, device="cuda", dtype=torch.float16)
    k = torch.randn_like(q); v = torch.randn_like(q)
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(iters):
        F.scaled_dot_product_attention(q,k,v,is_causal=False)
    torch.cuda.synchronize(); return time.time()-t0

def bench_flash(B=1,H=16,S=512,D=64, iters=50):
    q = torch.randn(B,H,S,D, device="cuda", dtype=torch.float16)
    k = torch.randn_like(q); v = torch.randn_like(q)
    torch.cuda.synchronize(); t0 = time.time()
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        for _ in range(iters):
            F.scaled_dot_product_attention(q,k,v,is_causal=False)
    torch.cuda.synchronize(); return time.time()-t0

print("default:", bench(), "s")
print("flash:  ", bench_flash(), "s")
