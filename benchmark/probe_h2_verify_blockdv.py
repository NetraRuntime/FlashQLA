# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]
"""Verify-first check of the H2-probe anomaly: the decode kernel with store_final_state=True is
~2x faster at block_DV=128 than the as-built block_DV=64. Before claiming a win, rule out
'fast-because-wrong': compare BOTH o and final_state of block_DV in {64,128} against decode_recur,
then re-time cleanly and isolate the K-major transposed ht-write cost (store_final True vs False).
"""
import sys
import torch

sys.path.insert(0, "/root/FlashQLA/tests")
from ref_gdr import decode_recur  # noqa: E402
from flash_qla.ops.gated_delta_rule.fused_recurrent.hopper.fused_recurrent_fwd import (  # noqa: E402
    tilelang_fused_recurrent_gdr_fwd,
)


def _build(H, block_DV, threads, store_final_state, dt):
    return tilelang_fused_recurrent_gdr_fwd(
        H, H, 128, 128, 128 ** -0.5,
        accum_dtype="float32", qkva_dtype=dt, g_dtype=torch.float32, b_dtype=torch.float32,
        h0_dtype=torch.float32, ht_dtype=torch.float32, o_dtype=dt, seqlen_dtype=torch.int32,
        use_initial_state=False, store_final_state=store_final_state, has_seqlens=False,
        block_DV=block_DV, threads=threads,
    )


def _run(kern, q, k, v, g, beta, B, H):
    o = torch.empty_like(v)
    ht = torch.empty(B, H, 128, 128, device="cuda", dtype=torch.float32)
    h0 = torch.empty(B, H, 128, 128, device="cuda", dtype=torch.float32)
    seqlens = torch.empty(B, dtype=torch.int32, device="cuda")
    kern(q, k, v, g, beta, h0, seqlens, o, ht)
    return o, ht


def _rel(a, b):
    return ((a.float() - b.float()).abs().max() / b.float().abs().max().clamp_min(1e-6)).item()


def correctness():
    print("=== CORRECTNESS (B=2,T=8,H=4) vs decode_recur ===")
    B, T, H = 2, 8, 4
    torch.manual_seed(0)
    from flash_qla.utils import l2norm
    q = l2norm(torch.randn(B, T, H, 128, device="cuda", dtype=torch.bfloat16))
    k = l2norm(torch.randn(B, T, H, 128, device="cuda", dtype=torch.bfloat16))
    v = torch.randn(B, T, H, 128, device="cuda", dtype=torch.bfloat16)
    g = torch.nn.functional.logsigmoid(torch.randn(B, T, H, device="cuda")) / 16
    beta = torch.randn(B, T, H, device="cuda").sigmoid()
    o_ref, s_ref = decode_recur(q, k, v, g, beta, scale=128 ** -0.5)
    for bdv in (64, 128):
        kern = _build(H, bdv, 128, True, q.dtype)
        o, ht = _run(kern, q, k, v, g, beta, B, H)
        oe, se = _rel(o, o_ref), _rel(ht, s_ref)
        tag = "OK" if (oe <= 0.02 and se <= 0.02) else "*** WRONG ***"
        print(f"  block_DV={bdv:<3d}: o_err={oe:.4f}  final_state_err={se:.4f}  [{tag}]")


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


def timing():
    print("\n=== TIMING (B=256,T=12,H=32) ===")
    B, T, H = 256, 12, 32
    torch.manual_seed(0)
    q = torch.randn(B, T, H, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, T, H, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(B, T, H, 128, device="cuda", dtype=torch.bfloat16)
    g = torch.nn.functional.logsigmoid(torch.randn(B, T, H, device="cuda")) / 16
    beta = torch.randn(B, T, H, device="cuda").sigmoid()
    for store in (True, False):
        print(f"  store_final_state={store}:")
        for bdv in (64, 128):
            kern = _build(H, bdv, 128, store, q.dtype)
            us = _time(lambda kern=kern: _run(kern, q, k, v, g, beta, B, H))
            print(f"    block_DV={bdv:<3d} threads=128  {us:8.1f} us")


if __name__ == "__main__":
    print(f"device: {torch.cuda.get_device_name()}")
    correctness()
    timing()
