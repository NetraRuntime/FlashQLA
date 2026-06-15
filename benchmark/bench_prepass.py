# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]
"""H1 crossover benchmark: the in-kernel-gated verify kernel (variant A) vs the dedup
prepass+host-gated path, end-to-end through recurrent_gated_delta_rule_verify(fuse_gating=True),
with prepass forced on/off. Eager event timing (conservative: the prepass's 2nd launch pays full
launch latency eagerly; under CUDA-graph replay it is cheaper, so an eager win is a real win).
Reports speedup A/prepass and whether the auto regime-gate (should_use_prepass) agrees with the
empirical winner -- used to calibrate PREPASS_MIN_T / PREPASS_CTA_FACTOR.
"""
import torch

from flash_qla import recurrent_gated_delta_rule_verify
from flash_qla.ops.gated_delta_rule.fused_recurrent import should_use_prepass
from flash_qla.ops.gated_delta_rule.fused_recurrent.hopper.fused_recurrent_verify import (
    PREPASS_MIN_T,
    PREPASS_MIN_WORK,
)


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


def _time_graph(fn, iters=100, warmup=20):
    # production path: capture into a CUDA graph, time replay (the 2nd-launch tax shrinks to a
    # graph node -- eager timing over-penalizes it). Warmup populates the persistent prepass scratch.
    st = torch.cuda.Stream()
    st.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(st)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    for _ in range(warmup):
        g.replay()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        g.replay()
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
    ibuf = torch.zeros(N + 1, T, Hv, 128, 128, dtype=torch.bfloat16, device="cuda")
    o = torch.empty(1, tot, Hv, 128, dtype=torch.bfloat16, device="cuda")
    return dict(A_log=A_log, a=a, dt_bias=dt_bias, q=q, k=k, v=v, b=b, ssm_states=pool,
                cache_indices=idx, query_start_loc=cu, intermediate_states_buffer=ibuf,
                intermediate_state_indices=idx, o=o)


def bench(N, T, Hk, Hv, tag):
    kw = _inputs(N, T, Hk, Hv)
    fA = lambda: recurrent_gated_delta_rule_verify(fuse_gating=True, prepass=False, **kw)
    fP = lambda: recurrent_gated_delta_rule_verify(fuse_gating=True, prepass=True, **kw)

    # build + parity (the two paths must agree within in-kernel-gating tolerance)
    oA = fA().clone()
    oP = fP().clone()
    err = ((oA.float() - oP.float()).abs().max() / oP.float().abs().max().clamp_min(1e-6)).item()

    tA = _time(fA)
    tP = _time(fP)
    sp = tA / tP if tP > 0 else 0.0
    auto = should_use_prepass(N, Hv, N * T)
    win = "WIN " if sp >= 1.05 else ("loss" if sp <= 0.97 else "neut")
    # MISCAL only if the gate picks the measurably-WRONG path: PP on a clear loss, or A on a clear win.
    # In the neutral band [0.97,1.05) either choice is fine (A preferred -> no extra launch).
    agree = "ok"
    if auto and sp <= 0.97:
        agree = "MISCAL"  # gate fired prepass but it regressed
    elif (not auto) and sp >= 1.05:
        agree = "MISCAL"  # gate kept A but prepass would have won
    print(f"  [{tag}] N={N:<4d} T={T:<2d} Hk={Hk:<2d} Hv={Hv:<2d}  "
          f"A={tA:8.1f}us  prepass={tP:8.1f}us  speedup={sp:4.2f}x [{win}]  "
          f"auto={'PP' if auto else 'A '} [{agree}]  parity={err:.4f}")


def bench_graph(N, T, Hk, Hv):
    # CUDA-graph (production) timing of variant A vs the prepass path, WITH gate calibration:
    # print work + the current gate's auto decision + MISCAL (gate picks the measurably-wrong path).
    kw = _inputs(N, T, Hk, Hv)
    fA = lambda: recurrent_gated_delta_rule_verify(fuse_gating=True, prepass=False, **kw)
    fP = lambda: recurrent_gated_delta_rule_verify(fuse_gating=True, prepass=True, **kw)
    fA(); fP()  # build kernels
    tA = _time_graph(fA)
    tP = _time_graph(fP)
    sp = tA / tP if tP > 0 else 0.0
    work = Hv * (N + N * T)
    auto = should_use_prepass(N, Hv, N * T)
    win = "WIN " if sp >= 1.05 else ("loss" if sp <= 0.97 else "neut")
    agree = "ok"
    if auto and sp <= 0.97:
        agree = "MISCAL"  # gate fired prepass but it regressed under graphs
    elif (not auto) and sp >= 1.05:
        agree = "MISCAL"  # gate kept A but prepass would have won under graphs
    print(f"  [graph] N={N:<4d} T={T:<2d} Hv={Hv:<2d} work={work:<6d}  "
          f"A={tA:8.1f}us  prepass={tP:8.1f}us  speedup={sp:4.2f}x [{win}]  "
          f"auto={'PP' if auto else 'A '} [{agree}]")


def main():
    sm = torch.cuda.get_device_properties().multi_processor_count
    print(f"device={torch.cuda.get_device_name()} SMs={sm} (TARGET_CTAS={int(sm*0.7)})")
    print(f"gate: PREPASS_MIN_T={PREPASS_MIN_T} PREPASS_MIN_WORK={PREPASS_MIN_WORK}  "
          f"(work=Hv*N*(1+T); prepass if t_avg>=MIN_T AND work>=MIN_WORK)\n")
    # PRODUCTION path is CUDA-graph: the prepass's 2nd launch shrinks to a graph node, so the
    # loss->win crossover sits at SMALLER work than the eager-calibrated gate assumes. Sweep the
    # small-N crossover zone under graphs to find where the prepass actually starts winning, and
    # flag where the current gate mis-fires. The N=1/T=12 floor (work=416) MUST stay loss/off.
    print("== CUDA-graph (production) crossover calibration, T in {4,8,12} (Hk=16,Hv=32) ==")
    for N in (1, 2, 4, 8, 16, 32, 64, 256):
        for T in (4, 8, 12):
            bench_graph(N, T, 16, 32)
        print()
    print("== T=1 (decode path) under graphs -- t_avg<MIN_T keeps prepass OFF; expect no win ==")
    for N in (8, 32, 128):
        bench_graph(N, 1, 16, 32)
    print("\n== N=1 single-request EAGER sanity (must remain A / loss) ==")
    for Hv in (64, 32, 16):
        bench(1, 12, max(1, Hv // 4), Hv, "lat")


if __name__ == "__main__":
    main()
