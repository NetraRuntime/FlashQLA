# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]
"""H1 verify-first probe (NO new kernel): measure the upper bound on what a gating+l2norm
pre-pass can save for the in-kernel-gated verify kernel.

Idea: the repo already ships two equivalent verify kernels --
  (A) in-kernel-gated  : recomputes g/beta + qk-l2norm INSIDE the hot loop, once per
                         (token, V-head, V-tile)  -> n_vt * grp redundancy for l2norm,
                         n_vt redundancy for gating.
  (B) host-gated       : reads PRE-computed g/beta/q_n/k_n; hot loop has NO transcendentals.

On identical raw inputs, time(A) - time(B) (with B's precompute done OUTSIDE the loop) is the
exact ceiling on H1's main-kernel saving. The realized H1 win = that ceiling minus the
(amortized-once) pre-pass cost. We also report the naive torch precompute cost for reference.
"""
import torch

from flash_qla.utils import l2norm
from flash_qla.ops.gated_delta_rule.fused_recurrent import gdn_sigmoid_gate
from flash_qla.ops.gated_delta_rule.fused_recurrent.hopper.fused_recurrent_verify import (
    fused_recurrent_gdr_verify_fwd,
    fused_recurrent_gdr_verify_gated_fwd,
)

PEAK_TBS = 3.35  # H100 HBM3


def _time(fn, iters=100, warmup=50):
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


def _inputs(N, T, Hk, Hv, seed=2025):
    torch.manual_seed(seed)
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
    return A_log, dt_bias, a, b, q, k, v, pool, cu, idx


def probe(N, T, Hk, Hv, tag):
    A_log, dt_bias, a, b, q, k, v, pool, cu, idx = _inputs(N, T, Hk, Hv)
    tot = N * T
    o = torch.empty(1, tot, Hv, 128, dtype=torch.bfloat16, device="cuda")
    ibuf = torch.zeros(N + 1, T, Hv, 128, 128, dtype=torch.bfloat16, device="cuda")

    # (B) host-gated precompute -- done ONCE, outside the timing loop (the H1 pre-pass stand-in)
    def precompute():
        g_ref, beta_ref = gdn_sigmoid_gate(A_log, a, dt_bias, b)
        return l2norm(q), l2norm(k), g_ref, beta_ref

    q_n, k_n, g_ref, beta_ref = precompute()

    fA = lambda: fused_recurrent_gdr_verify_gated_fwd(
        q, k, v, a, b, A_log, dt_bias, pool, idx, cu, ibuf, idx, o,
        scale=None, disable_state_update=True)
    fB = lambda: fused_recurrent_gdr_verify_fwd(
        q_n, k_n, v, g_ref, beta_ref, pool, idx, cu, ibuf, idx, o,
        scale=None, disable_state_update=True)

    # build + parity (A vs B must match within bf16 noise; same math, one in-kernel one host)
    fA()
    oA = o.clone()
    fB()
    oB = o.clone()
    err = ((oA.float() - oB.float()).abs().max() / oB.float().abs().max().clamp_min(1e-6)).item()

    tA = _time(fA)
    tB = _time(fB)
    tPre = _time(precompute, iters=50, warmup=20)  # naive torch pre-pass cost (reference)

    delta = tA - tB
    pct = 100.0 * delta / tA if tA > 0 else 0.0
    # ceiling speedup if pre-pass were free, and realistic if pre-pass == naive torch cost
    ceil_sp = tA / tB if tB > 0 else 0.0
    realistic_sp = tA / (tB + tPre) if (tB + tPre) > 0 else 0.0
    state_bytes = N * Hv * (1 + T) * 128 * 128 * 2
    gbsA = state_bytes / (tA * 1e-6) / 1e9
    block_DV = 64 if (N * Hv) * 2 >= int(torch.cuda.get_device_properties().multi_processor_count * 0.7) else 32
    n_vt = 128 // block_DV
    grp = Hv // Hk
    print(f"  [{tag}] N={N:<4d} T={T:<2d} Hk={Hk:<2d} Hv={Hv:<2d}  block_DV={block_DV} n_vt={n_vt} grp={grp}")
    print(f"        gated(A)={tA:8.1f}us  host(B)={tB:8.1f}us  delta={delta:7.1f}us ({pct:5.1f}% of A)  "
          f"ceil={ceil_sp:4.2f}x  torch_prepass={tPre:7.1f}us realistic={realistic_sp:4.2f}x  "
          f"A_bw={gbsA:5.0f}GB/s  parity={err:.4f}")


def main():
    sm = torch.cuda.get_device_properties().multi_processor_count
    print(f"device={torch.cuda.get_device_name()} SMs={sm} (TARGET_CTAS={int(sm*0.7)})")
    print("\n== latency-bound: single request, TP1..TP8 (N=1, T=12) ==")
    for Hv in (64, 32, 16, 8):
        probe(1, 12, max(1, Hv // 4), Hv, "lat")
    print("\n== latency-bound: small batch, short draft (N in {1,4}, T in {1,4}) ==")
    for N in (1, 4):
        for T in (1, 4):
            probe(N, T, 16, 32, "lat")
    print("\n== bandwidth-bound: server batched (Hv=32, Hk=16, N in {64,256}) ==")
    for N in (64, 256):
        for T in (1, 4, 12):
            probe(N, T, 16, 32, "bw")


if __name__ == "__main__":
    main()
