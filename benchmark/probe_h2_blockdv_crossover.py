# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]
"""Crossover for the decode-kernel store_final_state tile fix: block_DV=128 makes the K-major
transposed final-state write coalesced (~2x faster at large batch) but yields fewer CTAs (n_vt=1)
-> may be occupancy-starved at small batch. Sweep B (and H) to find where block_DV=128 beats the
as-built block_DV=64, so the dispatch can gate on it. store_final_state=True (the decode default)."""
import torch

from flash_qla.ops.gated_delta_rule.fused_recurrent.hopper.fused_recurrent_fwd import (
    tilelang_fused_recurrent_gdr_fwd,
)

SM = torch.cuda.get_device_properties().multi_processor_count
TARGET = int(SM * 0.7)


def _build(H, block_DV):
    return tilelang_fused_recurrent_gdr_fwd(
        H, H, 128, 128, 128 ** -0.5,
        accum_dtype="float32", qkva_dtype=torch.bfloat16, g_dtype=torch.float32, b_dtype=torch.float32,
        h0_dtype=torch.float32, ht_dtype=torch.float32, o_dtype=torch.bfloat16, seqlen_dtype=torch.int32,
        use_initial_state=False, store_final_state=True, has_seqlens=False,
        block_DV=block_DV, threads=128,
    )


def _time(fn, iters=50, warmup=25):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters * 1e3


def bench(B, T, H):
    torch.manual_seed(0)
    q = torch.randn(B, T, H, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, T, H, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(B, T, H, 128, device="cuda", dtype=torch.bfloat16)
    g = torch.nn.functional.logsigmoid(torch.randn(B, T, H, device="cuda")) / 16
    beta = torch.randn(B, T, H, device="cuda").sigmoid()
    h0 = torch.empty(B, H, 128, 128, device="cuda", dtype=torch.float32)
    ht = torch.empty(B, H, 128, 128, device="cuda", dtype=torch.float32)
    sl = torch.empty(B, dtype=torch.int32, device="cuda")
    o = torch.empty_like(v)
    res = {}
    for bdv in (64, 128):
        kern = _build(H, bdv)
        res[bdv] = _time(lambda kern=kern: kern(q, k, v, g, beta, h0, sl, o, ht))
    sp = res[64] / res[128]
    asbuilt = 64 if B * H * 2 >= TARGET else 32
    pick = "128" if sp >= 1.05 else ("64 " if sp <= 0.97 else "tie")
    print(f"  B={B:<4d} H={H:<3d} (B*H={B*H:<5d})  bdv64={res[64]:8.1f}us  bdv128={res[128]:8.1f}us  "
          f"128-speedup={sp:4.2f}x  best={pick}  (as-built picks {asbuilt})")


def main():
    print(f"device: {torch.cuda.get_device_name()} SMs={SM} TARGET={TARGET}\n")
    for H in (32, 16, 8):
        print(f"== H={H}, T=12, store_final_state=True ==")
        for B in (1, 2, 4, 8, 16, 32, 64, 128, 256):
            bench(B, 12, H)
        print()


if __name__ == "__main__":
    main()
