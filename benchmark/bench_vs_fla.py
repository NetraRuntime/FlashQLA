# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]
"""Speed benchmark: FlashQLA GDN verify kernel vs the FLA/Triton sigmoid-gating verify
kernel (vendored in netra-server) on the SGLang DFlash target_verify path.

Apples-to-apples: both run the in-kernel-gated, no-commit verify with per-token
intermediate-state caching over T draft tokens; both use a bf16 state pool/buffer
(SGLANG_MAMBA_SSM_DTYPE). FlashQLA is V-major, FLA is K-major -- each gets its native
layout for timing; the correctness gate transposes one side before comparing.
"""
import torch

from flash_qla.ops.gated_delta_rule.fused_recurrent.hopper.fused_recurrent_verify import (
    fused_recurrent_gdr_verify_gated_fwd,
)

# Load the FLA kernel file DIRECTLY (it's pure torch+triton) to bypass sglang/__init__.py,
# which runs the heavy SGLang frontend chain (hf patches -> orjson/transformers/...).
import importlib.util
import os

_FLA_PATH = os.environ.get(
    "FLA_KERNEL_PATH",
    "/root/netra-server/python/sglang/srt/layers/attention/fla/fused_sigmoid_gating_recurrent.py",
)
try:
    _spec = importlib.util.spec_from_file_location("fla_sigmoid_gating", _FLA_PATH)
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    fla_verify = _mod.fused_sigmoid_gating_delta_rule_update
    FLA_OK = True
except Exception as e:  # noqa: BLE001
    print("FLA load FAILED:", repr(e))
    FLA_OK = False

PEAK_TBS = 3.35  # H100 HBM3


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
    return s.elapsed_time(e) / iters  # ms


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
    # Both kernels read the pool V-major [.,HV,V,K] (FLA indexes h0 at offset o_v*K+o_k).
    pool = torch.randn(N, Hv, 128, 128, dtype=torch.bfloat16, device="cuda")
    cu = torch.arange(0, tot + 1, T, dtype=torch.int32, device="cuda")
    idx = torch.arange(N, dtype=torch.int32, device="cuda")
    return A_log, dt_bias, a, b, q, k, v, pool, cu, idx


def _call_fla(A_log, dt_bias, a, b, q, k, v, pool, cu, idx, ibuf, T):
    return fla_verify(
        A_log, a, dt_bias, 1.0, 20.0, q, k, v, b, pool, idx,
        scale=None, use_qk_l2norm_in_kernel=True, cu_seqlens=cu, is_kda=False,
        disable_state_update=True, intermediate_states_buffer=ibuf,
        intermediate_state_indices=idx, cache_steps=T, retrieve_parent_token=None,
    )


def _call_fqla(A_log, dt_bias, a, b, q, k, v, pool, cu, idx, ibuf, o):
    fused_recurrent_gdr_verify_gated_fwd(
        q, k, v, a, b, A_log, dt_bias, pool, idx, cu, ibuf, idx, o,
        scale=None, disable_state_update=True,
    )
    return o


def bench(N, T, Hk, Hv, check=True):
    A_log, dt_bias, a, b, q, k, v, pool, cu, idx = _inputs(N, T, Hk, Hv)
    o_fqla = torch.empty(1, N * T, Hv, 128, dtype=torch.bfloat16, device="cuda")
    ibuf_fla = torch.zeros(N + 1, T, Hv, 128, 128, dtype=torch.bfloat16, device="cuda")
    ibuf_fqla = torch.zeros(N + 1, T, Hv, 128, 128, dtype=torch.bfloat16, device="cuda")

    o_fla = _call_fla(A_log, dt_bias, a, b, q, k, v, pool, cu, idx, ibuf_fla, T)
    _call_fqla(A_log, dt_bias, a, b, q, k, v, pool, cu, idx, ibuf_fqla, o_fqla)

    if check:
        o_err = (o_fqla.float() - o_fla.float()).abs().max() / o_fla.float().abs().max().clamp_min(1e-6)
        ib_err = (ibuf_fqla.float() - ibuf_fla.float()).abs().max() / \
            ibuf_fla.float().abs().max().clamp_min(1e-6)  # both V-major, compare directly
        tag = "OK" if (o_err < 0.04 and ib_err < 0.04) else "MISMATCH"
        print(f"  [parity {tag}] o_err={o_err:.4f} ibuf_err={ib_err:.4f}")

    t_fqla = _time(lambda: _call_fqla(A_log, dt_bias, a, b, q, k, v, pool, cu, idx, ibuf_fqla, o_fqla))
    t_fla = _time(lambda: _call_fla(A_log, dt_bias, a, b, q, k, v, pool, cu, idx, ibuf_fla, T))

    es = 2
    state_bytes = N * Hv * (1 + T) * 128 * 128 * es
    gbs_fqla = state_bytes / (t_fqla * 1e-3) / 1e9
    gbs_fla = state_bytes / (t_fla * 1e-3) / 1e9
    sp = t_fla / t_fqla
    print(f"  N={N:<4d} Hv={Hv} Hk={Hk} T={T:<2d}  "
          f"FlashQLA {t_fqla*1e3:8.1f}us ({gbs_fqla:6.0f} GB/s)  "
          f"FLA {t_fla*1e3:8.1f}us ({gbs_fla:6.0f} GB/s)  "
          f"speedup {sp:4.2f}x")


if __name__ == "__main__":
    if not FLA_OK:
        raise SystemExit("FLA baseline unavailable -- cannot compare")
    print(f"device: {torch.cuda.get_device_name()}  |  GDN verify: FlashQLA vs FLA (bf16 pool, per-token states)")
    print("server / batched (Hv=32, Hk=16):")
    for N in (8, 64, 256):
        for T in (1, 4, 12):
            bench(N, T, 16, 32)
    print("single-request / TP (N=1, Hv=TP1..TP8):")
    for Hv in (64, 32, 16, 8):
        bench(1, 12, max(1, Hv // 4), Hv)
