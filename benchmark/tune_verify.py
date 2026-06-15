# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]
"""Autotune the gemm-free verify kernel: sweep (block_DV, threads) for the bandwidth-bound
large-batch regime where FLA currently edges ahead. Memory-bound -> occupancy (smaller tiles,
more CTAs) usually beats fewer-but-bigger tiles. Reports achieved HBM GB/s per config."""
import itertools

import torch

from flash_qla.ops.gated_delta_rule.fused_recurrent.hopper.fused_recurrent_verify import (
    tilelang_fused_recurrent_gdr_verify_gated,
)

PEAK_TBS = 3.35


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
    return s.elapsed_time(e) / iters


def sweep(N, T, Hk, Hv):
    torch.manual_seed(0)
    tot = N * T
    A_log = torch.randn(Hv, dtype=torch.float32, device="cuda")
    dt_bias = torch.randn(Hv, dtype=torch.float32, device="cuda")
    a = torch.randn(1, tot, Hv, dtype=torch.bfloat16, device="cuda")
    b = torch.randn(1, tot, Hv, dtype=torch.bfloat16, device="cuda")
    q = torch.randn(1, tot, Hk, 128, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(1, tot, Hk, 128, dtype=torch.bfloat16, device="cuda")
    v = torch.randn(1, tot, Hv, 128, dtype=torch.bfloat16, device="cuda")
    pool = torch.randn(N, Hv, 128, 128, dtype=torch.bfloat16, device="cuda")
    cu = torch.arange(0, tot + 1, T, dtype=torch.int32, device="cuda")
    idx = torch.arange(N, dtype=torch.int32, device="cuda")
    o = torch.empty(1, tot, Hv, 128, dtype=torch.bfloat16, device="cuda")
    ibuf = torch.zeros(N + 1, T, Hv, 128, 128, dtype=torch.bfloat16, device="cuda")
    state_bytes = N * Hv * (1 + T) * 128 * 128 * 2

    print(f"\n== N={N} T={T} Hv={Hv} Hk={Hk}  (state {state_bytes/1e6:.0f} MB) ==")
    best = None
    for block_DV, threads in itertools.product([32, 64, 128], [64, 128, 256, 512]):
        if threads < block_DV:  # need enough threads for the [block_DV,DK] work
            continue
        try:
            kern = tilelang_fused_recurrent_gdr_verify_gated(
                Hv, Hk, 128, 128, 128 ** -0.5,
                accum_dtype="float32", qkva_dtype=q.dtype, ab_dtype=a.dtype, gate_dtype=A_log.dtype,
                pool_dtype=pool.dtype, o_dtype=o.dtype, seqlen_dtype=cu.dtype, idx_dtype=idx.dtype,
                store_intermediate=True, disable_state_update=True,
                block_DV=block_DV, threads=threads,
            )
            fn = lambda kern=kern: kern(q, k, v, a, b, A_log, dt_bias, pool, idx, cu, idx, o, ibuf)
            ms = _time(fn)
        except Exception as ex:  # noqa: BLE001
            print(f"  block_DV={block_DV:<3d} threads={threads:<3d}  FAIL {str(ex).splitlines()[-1][:50]}")
            continue
        gbs = state_bytes / (ms * 1e-3) / 1e9
        marker = ""
        if best is None or ms < best[0]:
            best = (ms, block_DV, threads)
            marker = " <-- best"
        print(f"  block_DV={block_DV:<3d} threads={threads:<3d}  {ms*1e3:8.1f} us  {gbs:7.0f} GB/s ({100*gbs/1000/PEAK_TBS:4.1f}%){marker}")
    print(f"  BEST: block_DV={best[1]} threads={best[2]}  {best[0]*1e3:.1f} us")


if __name__ == "__main__":
    print(f"device: {torch.cuda.get_device_name()}")
    for (N, T, Hk, Hv) in [(256, 12, 16, 32), (64, 12, 16, 32), (256, 4, 16, 32), (8, 1, 16, 32)]:
        sweep(N, T, Hk, Hv)
