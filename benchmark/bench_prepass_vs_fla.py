# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]
"""High-batch 3-way: FLA/Triton sigmoid-gating verify vs FlashQLA variant-A (in-kernel-gated) vs
FlashQLA H1 prepass path -- all in ONE harness, parity-gated, same inputs. Answers "how much vs
FLA at high batch now that H1 dedups the in-loop gating/l2norm". Event timing (at N>=64,T>=12 the
~15us launch is <1% of the 100us-1.6ms runtime, so eager is reliable+fair, matching bench_vs_fla)."""
import importlib.util
import os
import torch

from flash_qla import recurrent_gated_delta_rule_verify

_FLA_PATH = os.environ.get(
    "FLA_KERNEL_PATH",
    "/root/netra-server/python/sglang/srt/layers/attention/fla/fused_sigmoid_gating_recurrent.py",
)
_spec = importlib.util.spec_from_file_location("fla_sigmoid_gating", _FLA_PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
fla_verify = _mod.fused_sigmoid_gating_delta_rule_update

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


def bench(N, T, Hk, Hv):
    A_log, dt_bias, a, b, q, k, v, pool, cu, idx = _inputs(N, T, Hk, Hv)
    tot = N * T
    o = torch.empty(1, tot, Hv, 128, dtype=torch.bfloat16, device="cuda")
    ib_fla = torch.zeros(N + 1, T, Hv, 128, 128, dtype=torch.bfloat16, device="cuda")
    ib_a = torch.zeros(N + 1, T, Hv, 128, 128, dtype=torch.bfloat16, device="cuda")
    ib_pp = torch.zeros(N + 1, T, Hv, 128, 128, dtype=torch.bfloat16, device="cuda")

    fla = lambda: fla_verify(
        A_log, a, dt_bias, 1.0, 20.0, q, k, v, b, pool, idx, scale=None,
        use_qk_l2norm_in_kernel=True, cu_seqlens=cu, is_kda=False, disable_state_update=True,
        intermediate_states_buffer=ib_fla, intermediate_state_indices=idx, cache_steps=T,
        retrieve_parent_token=None)
    kw = dict(ssm_states=pool, cache_indices=idx, query_start_loc=cu,
              intermediate_state_indices=idx, o=o, fuse_gating=True, disable_state_update=True)
    varA = lambda: recurrent_gated_delta_rule_verify(
        A_log, a, dt_bias, q, k, v, b, intermediate_states_buffer=ib_a, prepass=False, **kw)
    pp = lambda: recurrent_gated_delta_rule_verify(
        A_log, a, dt_bias, q, k, v, b, intermediate_states_buffer=ib_pp, prepass=True, **kw)

    o_fla = fla().clone()
    varA(); o_a = o.clone()
    pp(); o_pp = o.clone()
    er = lambda x, y: ((x.float() - y.float()).abs().max() / y.float().abs().max().clamp_min(1e-6)).item()
    parity = f"A:{er(o_a, o_fla):.4f} PP:{er(o_pp, o_fla):.4f}"

    t_fla, t_a, t_pp = _time(fla), _time(varA), _time(pp)
    sb = N * Hv * (1 + T) * 128 * 128 * 2
    print(f"  N={N:<4d} T={T:<2d} Hk={Hk} Hv={Hv}  "
          f"FLA {t_fla:8.1f}us ({sb/(t_fla*1e-6)/1e9:5.0f}GB/s)  "
          f"varA {t_a:8.1f}us ({t_fla/t_a:4.2f}x)  "
          f"prepass {t_pp:8.1f}us ({t_fla/t_pp:4.2f}x vs FLA, {t_a/t_pp:4.2f}x vs A)  [parity o {parity}]")


def main():
    print(f"device: {torch.cuda.get_device_name()}  |  GDN verify high-batch: FLA vs FlashQLA varA vs prepass")
    for N in (64, 128, 256):
        for T in (4, 12):
            bench(N, T, 16, 32)


if __name__ == "__main__":
    main()
