# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]
"""Sweep step 1 -- attribution: where does time go in the post-H1 verify main kernel (host-gated)
and the decode kernel? Isolate the per-token ibuf write (store_intermediate on/off) and the pool
commit, to decide the next lever: if the ibuf write is a large TIME fraction -> DRAM/store-bound
(vectorize the store); if small -> compute-bound on the recurrence (fuse the elementwise passes).
g/beta/q_n/k_n are precomputed host-side (outside timing) so this is the post-H1 main kernel only.
"""
import torch

from flash_qla.utils import l2norm
from flash_qla.ops.gated_delta_rule.fused_recurrent import gdn_sigmoid_gate
from flash_qla.ops.gated_delta_rule.fused_recurrent.hopper.fused_recurrent_verify import (
    fused_recurrent_gdr_verify_fwd,
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
    return s.elapsed_time(e) / iters * 1e3  # us


def attrib(N, T, Hk, Hv):
    torch.manual_seed(0)
    tot = N * T
    A_log = torch.randn(Hv, device="cuda")
    dt_bias = torch.randn(Hv, device="cuda")
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
    g, beta = gdn_sigmoid_gate(A_log, a, dt_bias, b)
    qn, kn = l2norm(q), l2norm(k)

    full = lambda: fused_recurrent_gdr_verify_fwd(
        qn, kn, v, g, beta, pool, idx, cu, ibuf, idx, o, disable_state_update=True)
    noibuf = lambda: fused_recurrent_gdr_verify_fwd(
        qn, kn, v, g, beta, pool, idx, cu, None, idx, o, disable_state_update=True)
    commit = lambda: fused_recurrent_gdr_verify_fwd(
        qn, kn, v, g, beta, pool, idx, cu, ibuf, idx, o, disable_state_update=False)

    t_full, t_no, t_commit = _time(full), _time(noibuf), _time(commit)
    ibuf_bytes = N * Hv * T * 128 * 128 * 2
    ibuf_us = t_full - t_no
    print(f"  N={N:<4d} T={T:<2d} Hv={Hv}  full={t_full:8.1f}us  no-ibuf={t_no:8.1f}us  "
          f"ibuf-write={ibuf_us:7.1f}us ({100*ibuf_us/t_full:4.1f}% of time, "
          f"{ibuf_bytes/(ibuf_us*1e-6)/1e9 if ibuf_us>0 else 0:5.0f}GB/s)  "
          f"+commit={t_commit-t_full:5.1f}us")


def main():
    print(f"device: {torch.cuda.get_device_name()}  |  verify host-gated (post-H1) time attribution")
    for N in (64, 256):
        for T in (4, 12):
            attrib(N, T, 16, 32)


if __name__ == "__main__":
    main()
