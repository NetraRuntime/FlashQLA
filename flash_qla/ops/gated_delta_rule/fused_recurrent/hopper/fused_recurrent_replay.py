# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]
"""SGLang DFlash reduced-cache REPLAY kernel (K0): recompute the accepted-tail GDN state
from the pre-verify committed state and commit the final accepted state back to the pool.

This is the state-only counterpart of the verify kernel. After a no-commit target verify
(`disable_state_update=True`) caches ZERO intermediate states (K0), the pool slot still
holds the PRE-verify state. Once the target accepts `accept_len` tokens, replay re-runs the
recurrence over the first `accept_len` draft tokens of the block from that pre-verify state
and commits the resulting state -- exactly what the next decode/verify step must read.

PARITY-BY-CONSTRUCTION: the per-token recurrence body (k-l2norm + sigmoid gating + delta
state update, fp32 accumulate, exp2 decay) is COPIED VERBATIM from
`tilelang_fused_recurrent_gdr_verify_gated`. The committed state therefore equals the
intermediate state the verify kernel passed through at step `accept_len`, bit-for-bit.
The q-projection / output `o` of verify is DROPPED here: it feeds the verify logits only,
never the state `S`, so omitting it leaves the committed state identical (and is cheaper).

ADDRESSING (matches the GDNKernelDispatcher.state_update contract used by the Triton replay):
  for replay row bb:
    req   = input_sequence_indices[bb]          # which request's block
    slot_in  = initial_state_indices[bb]        # read pre-verify state (K0: == destination)
    slot_out = cache_indices[bb]                # commit final accepted state
    tail  = input_sequence_lengths[bb]          # accept_len (= target_steps[bb]+1 for K0)
    token tt = req * input_token_stride + input_token_start + t,  t in [0, tail)
`input_token_start` (0 for K0) and `input_token_stride` (= draft_token_num) are compile-time
constants (stable per deployment) -> capture-safe, no host sync, all buffers caller-provided.
"""
import torch
import tilelang
import tilelang.language as T

MULTI_PROCESSOR_COUNT = torch.cuda.get_device_properties().multi_processor_count
TARGET_NUM_CTAS = int(MULTI_PROCESSOR_COUNT * 0.7)


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True})
def tilelang_fused_recurrent_gdr_replay(
    H,
    Hg,
    DK,
    DV,
    input_token_start,
    input_token_stride,
    accum_dtype,
    qkva_dtype,
    ab_dtype,
    gate_dtype,
    pool_dtype,
    seqlen_dtype,
    idx_dtype,
    l2norm_eps=1e-6,
    softplus_thr=20.0,
    allow_neg_eigval=False,
    block_DV=128,
    threads=128,
):
    """K0 state-only replay. In-kernel fused gating (g=-exp(A_log)*softplus(a+dt_bias),
    beta=sigmoid(b)) + k-l2norm, identical to the gated verify kernel; no q-proj / no output."""
    total_tokens = T.dynamic("total_tokens")
    N = T.dynamic("N")  # number of replay rows (subset of requests)
    num_slots = T.dynamic("num_slots")
    qk_shape = (1, total_tokens, Hg, DK)
    v_shape = (1, total_tokens, H, DV)
    ab_shape = (1, total_tokens, H)
    pool_shape = (num_slots, H, DV, DK)  # V-major [., H, V, K] (SGLang contract)
    n_vt = (DV + block_DV - 1) // block_DV
    beta_mul = 2.0 if allow_neg_eigval else 1.0

    @T.prim_func
    def kernel(
        k: T.Tensor(qk_shape, qkva_dtype),
        v: T.Tensor(v_shape, qkva_dtype),
        a: T.Tensor(ab_shape, ab_dtype),
        b: T.Tensor(ab_shape, ab_dtype),
        A_log: T.Tensor([H], gate_dtype),
        dt_bias: T.Tensor([H], gate_dtype),
        pool: T.Tensor(pool_shape, pool_dtype),
        initial_state_indices: T.Tensor([N], idx_dtype),
        cache_indices: T.Tensor([N], idx_dtype),
        input_sequence_indices: T.Tensor([N], idx_dtype),
        input_sequence_lengths: T.Tensor([N], seqlen_dtype),
    ):
        with T.Kernel(n_vt * N * H, threads=threads) as (bbhv,):
            bbh = bbhv // n_vt
            bv = bbhv % n_vt
            bb = bbh // H  # replay row index
            bh = bbh % H
            bhg = bh // (H // Hg)
            v0 = bv * block_DV

            slot_in = T.alloc_var("int32")
            slot_out = T.alloc_var("int32")
            req = T.alloc_var("int32")
            tail = T.alloc_var("int32")
            seq_start = T.alloc_var("int32")
            slot_in = initial_state_indices[bb]
            slot_out = cache_indices[bb]
            req = input_sequence_indices[bb]
            tail = input_sequence_lengths[bb]
            seq_start = req * input_token_stride + input_token_start
            a_log_h = T.alloc_var(accum_dtype)
            dt_b_h = T.alloc_var(accum_dtype)
            a_log_h = A_log[bh]
            dt_b_h = dt_bias[bh]

            S = T.alloc_fragment((block_DV, DK), accum_dtype)
            prod = T.alloc_fragment((block_DV, DK), accum_dtype)
            k_s = T.alloc_shared((1, DK), qkva_dtype)
            k_n = T.alloc_shared((1, DK), qkva_dtype)
            v_s = T.alloc_shared((1, block_DV), qkva_dtype)
            ksq = T.alloc_fragment((1, DK), accum_dtype)
            ssq = T.alloc_fragment((1,), accum_dtype)
            kS = T.alloc_fragment((block_DV,), accum_dtype)
            vnew = T.alloc_fragment((block_DV,), accum_dtype)
            decay = T.alloc_fragment((1,), accum_dtype)
            bt = T.alloc_fragment((1,), accum_dtype)

            # gather pre-verify state from the READ slot
            T.clear(S)
            with T.If(slot_in >= 0):
                with T.Then():
                    for j_v, j_k in T.Parallel(block_DV, DK):
                        S[j_v, j_k] = pool[slot_in, bh, v0 + j_v, j_k]

            for t in T.serial(tail):
                tt = seq_start + t
                T.copy(k[0, tt : tt + 1, bhg, 0:DK], k_s)
                T.copy(v[0, tt : tt + 1, bh, v0 : v0 + block_DV], v_s)
                # in-kernel k l2norm (q-norm dropped: q only feeds o, not S)
                for _i, j_k in T.Parallel(1, DK):
                    ksq[0, j_k] = k_s[0, j_k] * k_s[0, j_k]
                T.reduce_sum(ksq, ssq, dim=1)
                for _i, j_k in T.Parallel(1, DK):
                    k_n[0, j_k] = k_s[0, j_k] * T.rsqrt(ssq[0] + l2norm_eps)
                # in-kernel gating: g = -exp(A_log)*softplus(a+dt_bias); beta = sigmoid(b)
                x = a[0, tt, bh] + dt_b_h
                sp = T.if_then_else(x > softplus_thr, x, T.log(1.0 + T.exp(x)))
                decay[0] = T.exp2((-T.exp(a_log_h) * sp) * 1.442695)
                bt[0] = beta_mul * T.sigmoid(b[0, tt, bh])
                for j_v, j_k in T.Parallel(block_DV, DK):
                    S[j_v, j_k] *= decay[0]
                for j_v, j_k in T.Parallel(block_DV, DK):
                    prod[j_v, j_k] = k_n[0, j_k] * S[j_v, j_k]
                T.reduce_sum(prod, kS, dim=1)
                for j_v in T.Parallel(block_DV):
                    vnew[j_v] = bt[0] * (v_s[0, j_v] - kS[j_v])
                for j_v, j_k in T.Parallel(block_DV, DK):
                    S[j_v, j_k] += k_n[0, j_k] * vnew[j_v]

            # commit final accepted state to the WRITE slot
            with T.If(slot_out >= 0):
                with T.Then():
                    for j_v, j_k in T.Parallel(block_DV, DK):
                        pool[slot_out, bh, v0 + j_v, j_k] = S[j_v, j_k]

    return kernel


def fused_recurrent_gdr_replay_fwd(
    k,
    v,
    a,
    b,
    A_log,
    dt_bias,
    pool,
    initial_state_indices,
    cache_indices,
    input_sequence_indices,
    input_sequence_lengths,
    input_token_start,
    input_token_stride,
    allow_neg_eigval=False,
):
    """K0 reduced-cache replay entry. Raw a,b,A_log,dt_bias (in-kernel gating); k/v contiguous
    [1, total_tokens, H*, 128]; pool V-major [num_slots, Hv, 128, 128]. All buffers
    caller-provided (graph-safe). Commits the accepted-tail final state in place; returns None."""
    _, total_tokens, Hg, K = k.shape
    _, _, H, V = v.shape
    N = cache_indices.shape[0]
    assert K == V == 128 and H % Hg == 0

    grid_base = N * H
    block_DV = 64 if grid_base * 2 >= TARGET_NUM_CTAS else 32

    kern = tilelang_fused_recurrent_gdr_replay(
        H,
        Hg,
        K,
        V,
        int(input_token_start),
        int(input_token_stride),
        accum_dtype="float32",
        qkva_dtype=k.dtype,
        ab_dtype=a.dtype,
        gate_dtype=A_log.dtype,
        pool_dtype=pool.dtype,
        seqlen_dtype=input_sequence_lengths.dtype,
        idx_dtype=cache_indices.dtype,
        allow_neg_eigval=allow_neg_eigval,
        block_DV=block_DV,
        threads=max(128, block_DV * 2),
    )
    kern(
        k,
        v,
        a,
        b,
        A_log,
        dt_bias,
        pool,
        initial_state_indices,
        cache_indices,
        input_sequence_indices,
        input_sequence_lengths,
    )
    return None
