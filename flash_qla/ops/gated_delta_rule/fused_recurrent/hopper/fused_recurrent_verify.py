# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]
"""SGLang verify kernel: gemm-free GDN recurrence + paged V-major (bf16) state pool,
per-token intermediate states, no-commit, varlen cu_seqlens. Host-side gating (g/beta
pre-activated, q/k pre-l2normed by the wrapper). CUDA-graph safe (no host sync / no alloc
in the captured entry; all buffers caller-provided)."""
import torch
import tilelang
import tilelang.language as T

MULTI_PROCESSOR_COUNT = torch.cuda.get_device_properties().multi_processor_count
TARGET_NUM_CTAS = int(MULTI_PROCESSOR_COUNT * 0.7)


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True})
def tilelang_fused_recurrent_gdr_verify(
    H,
    Hg,
    DK,
    DV,
    scale,
    accum_dtype,
    qkva_dtype,
    g_dtype,
    b_dtype,
    pool_dtype,
    o_dtype,
    seqlen_dtype,
    idx_dtype,
    store_intermediate,
    disable_state_update,
    block_DV=128,
    threads=128,
):
    total_tokens = T.dynamic("total_tokens")
    N = T.dynamic("N")  # number of requests
    num_slots = T.dynamic("num_slots")
    num_cache_slots = T.dynamic("num_cache_slots")
    cache_steps = T.dynamic("cache_steps")
    q_shape = (1, total_tokens, Hg, DK)
    v_shape = (1, total_tokens, H, DV)
    g_shape = (1, total_tokens, H)
    pool_shape = (num_slots, H, DV, DK)  # V-major [., H, V, K]
    ibuf_shape = (num_cache_slots, cache_steps, H, DV, DK)  # V-major
    n_vt = (DV + block_DV - 1) // block_DV

    @T.prim_func
    def kernel(
        q: T.Tensor(q_shape, qkva_dtype),
        k: T.Tensor(q_shape, qkva_dtype),
        v: T.Tensor(v_shape, qkva_dtype),
        g: T.Tensor(g_shape, g_dtype),
        b: T.Tensor(g_shape, b_dtype),
        pool: T.Tensor(pool_shape, pool_dtype),
        state_indices: T.Tensor([N], idx_dtype),
        cu_seqlens: T.Tensor([N + 1], seqlen_dtype),
        intermediate_state_indices: T.Tensor([N], idx_dtype),
        o: T.Tensor(v_shape, o_dtype),
        ibuf: T.Tensor(ibuf_shape, pool_dtype),
    ):
        with T.Kernel(n_vt * N * H, threads=threads) as (bbhv,):
            bbh = bbhv // n_vt
            bv = bbhv % n_vt
            bb = bbh // H  # request index
            bh = bbh % H
            bhg = bh // (H // Hg)
            v0 = bv * block_DV

            slot = T.alloc_var("int32")
            cslot = T.alloc_var("int32")
            seq_start = T.alloc_var("int32")
            seq_end = T.alloc_var("int32")
            slot = state_indices[bb]
            cslot = intermediate_state_indices[bb]
            seq_start = cu_seqlens[bb]
            seq_end = cu_seqlens[bb + 1]

            S = T.alloc_fragment((block_DV, DK), accum_dtype)  # state [V-tile, K] fp32
            prod = T.alloc_fragment((block_DV, DK), accum_dtype)
            q_s = T.alloc_shared((1, DK), qkva_dtype)
            k_s = T.alloc_shared((1, DK), qkva_dtype)
            v_s = T.alloc_shared((1, block_DV), qkva_dtype)
            o_sh = T.alloc_shared((1, block_DV), o_dtype)
            kS = T.alloc_fragment((block_DV,), accum_dtype)
            oo = T.alloc_fragment((block_DV,), accum_dtype)
            vnew = T.alloc_fragment((block_DV,), accum_dtype)
            decay = T.alloc_fragment((1,), accum_dtype)
            bt = T.alloc_fragment((1,), accum_dtype)

            # gather V-major pool[slot, bh, v0:v0+block_DV, :] directly into S[v, dk]
            T.clear(S)
            with T.If(slot >= 0):
                with T.Then():
                    for j_v, j_k in T.Parallel(block_DV, DK):
                        S[j_v, j_k] = pool[slot, bh, v0 + j_v, j_k]

            for t in T.serial(seq_end - seq_start):
                tt = seq_start + t  # absolute token position in the flattened layout
                T.copy(q[0, tt : tt + 1, bhg, 0:DK], q_s)
                T.copy(k[0, tt : tt + 1, bhg, 0:DK], k_s)
                T.copy(v[0, tt : tt + 1, bh, v0 : v0 + block_DV], v_s)
                decay[0] = T.exp2(g[0, tt, bh] * 1.442695)
                bt[0] = b[0, tt, bh]
                for j_v, j_k in T.Parallel(block_DV, DK):
                    S[j_v, j_k] *= decay[0]
                for j_v, j_k in T.Parallel(block_DV, DK):
                    prod[j_v, j_k] = k_s[0, j_k] * S[j_v, j_k]
                T.reduce_sum(prod, kS, dim=1)
                for j_v in T.Parallel(block_DV):
                    vnew[j_v] = bt[0] * (v_s[0, j_v] - kS[j_v])
                for j_v, j_k in T.Parallel(block_DV, DK):
                    S[j_v, j_k] += k_s[0, j_k] * vnew[j_v]
                for j_v, j_k in T.Parallel(block_DV, DK):
                    prod[j_v, j_k] = q_s[0, j_k] * S[j_v, j_k]
                T.reduce_sum(prod, oo, dim=1)
                for j_v in T.Parallel(block_DV):
                    o_sh[0, j_v] = oo[j_v] * scale
                T.copy(o_sh, o[0, tt : tt + 1, bh, v0 : v0 + block_DV])
                # per-token intermediate (V-major), gated by the POOL slot mask
                if store_intermediate:
                    with T.If(slot >= 0):
                        with T.Then():
                            for j_v, j_k in T.Parallel(block_DV, DK):
                                ibuf[cslot, t, bh, v0 + j_v, j_k] = S[j_v, j_k]

            # commit final state to the pool unless no-commit (verify)
            if not disable_state_update:
                with T.If(slot >= 0):
                    with T.Then():
                        for j_v, j_k in T.Parallel(block_DV, DK):
                            pool[slot, bh, v0 + j_v, j_k] = S[j_v, j_k]

    return kernel


def fused_recurrent_gdr_verify_fwd(
    q,
    k,
    v,
    g,
    beta,
    pool,
    state_indices,
    cu_seqlens,
    intermediate_states_buffer,
    intermediate_state_indices,
    o,
    scale=None,
    disable_state_update=True,
):
    """Graph-safe low-level verify entry. ALL buffers caller-preallocated (o, pool, ibuf);
    no host sync, no allocation. g/beta pre-activated and q/k pre-l2normed host-side."""
    _, total_tokens, Hg, K = k.shape
    _, _, H, V = v.shape
    N = state_indices.shape[0]
    assert K == V == 128 and H % Hg == 0
    scale = scale or K ** -0.5
    store_intermediate = intermediate_states_buffer is not None

    # block_DV=64 (2 V-tiles) @ threads=128 is the bandwidth sweet spot (autotuned, H100);
    # 32 (4 V-tiles) for the low-CTA tail. block_DV=128 is occupancy-starved -> never used.
    grid_base = N * H
    block_DV = 64 if grid_base * 2 >= TARGET_NUM_CTAS else 32

    kern = tilelang_fused_recurrent_gdr_verify(
        H,
        Hg,
        K,
        V,
        scale,
        accum_dtype="float32",
        qkva_dtype=q.dtype,
        g_dtype=g.dtype,
        b_dtype=beta.dtype,
        pool_dtype=pool.dtype,
        o_dtype=o.dtype,
        seqlen_dtype=cu_seqlens.dtype,
        idx_dtype=state_indices.dtype,
        store_intermediate=store_intermediate,
        disable_state_update=disable_state_update,
        block_DV=block_DV,
        threads=128,
    )
    kern(q, k, v, g, beta, pool, state_indices, cu_seqlens,
         intermediate_state_indices, o, intermediate_states_buffer)
    return o


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True})
def tilelang_fused_recurrent_gdr_verify_gated(
    H, Hg, DK, DV, scale, accum_dtype, qkva_dtype, ab_dtype, gate_dtype, pool_dtype,
    o_dtype, seqlen_dtype, idx_dtype, store_intermediate, disable_state_update,
    l2norm_eps=1e-6, softplus_thr=20.0, allow_neg_eigval=False, block_DV=128, threads=128,
):
    """In-kernel fused-gating verify kernel (req #5): takes raw a,b,A_log,dt_bias and
    computes g=-exp(A_log)*softplus(a+dt_bias), beta=sigmoid(b), and qk-l2norm in-kernel."""
    total_tokens = T.dynamic("total_tokens")
    N = T.dynamic("N")
    num_slots = T.dynamic("num_slots")
    num_cache_slots = T.dynamic("num_cache_slots")
    cache_steps = T.dynamic("cache_steps")
    qk_shape = (1, total_tokens, Hg, DK)
    v_shape = (1, total_tokens, H, DV)
    ab_shape = (1, total_tokens, H)
    pool_shape = (num_slots, H, DV, DK)
    ibuf_shape = (num_cache_slots, cache_steps, H, DV, DK)
    n_vt = (DV + block_DV - 1) // block_DV
    beta_mul = 2.0 if allow_neg_eigval else 1.0

    @T.prim_func
    def kernel(
        q: T.Tensor(qk_shape, qkva_dtype),
        k: T.Tensor(qk_shape, qkva_dtype),
        v: T.Tensor(v_shape, qkva_dtype),
        a: T.Tensor(ab_shape, ab_dtype),
        b: T.Tensor(ab_shape, ab_dtype),
        A_log: T.Tensor([H], gate_dtype),
        dt_bias: T.Tensor([H], gate_dtype),
        pool: T.Tensor(pool_shape, pool_dtype),
        state_indices: T.Tensor([N], idx_dtype),
        cu_seqlens: T.Tensor([N + 1], seqlen_dtype),
        intermediate_state_indices: T.Tensor([N], idx_dtype),
        o: T.Tensor(v_shape, o_dtype),
        ibuf: T.Tensor(ibuf_shape, pool_dtype),
    ):
        with T.Kernel(n_vt * N * H, threads=threads) as (bbhv,):
            bbh = bbhv // n_vt
            bv = bbhv % n_vt
            bb = bbh // H
            bh = bbh % H
            bhg = bh // (H // Hg)
            v0 = bv * block_DV

            slot = T.alloc_var("int32")
            cslot = T.alloc_var("int32")
            seq_start = T.alloc_var("int32")
            seq_end = T.alloc_var("int32")
            slot = state_indices[bb]
            cslot = intermediate_state_indices[bb]
            seq_start = cu_seqlens[bb]
            seq_end = cu_seqlens[bb + 1]
            a_log_h = T.alloc_var(accum_dtype)
            dt_b_h = T.alloc_var(accum_dtype)
            a_log_h = A_log[bh]
            dt_b_h = dt_bias[bh]

            S = T.alloc_fragment((block_DV, DK), accum_dtype)
            prod = T.alloc_fragment((block_DV, DK), accum_dtype)
            q_s = T.alloc_shared((1, DK), qkva_dtype)
            k_s = T.alloc_shared((1, DK), qkva_dtype)
            q_n = T.alloc_shared((1, DK), qkva_dtype)
            k_n = T.alloc_shared((1, DK), qkva_dtype)
            v_s = T.alloc_shared((1, block_DV), qkva_dtype)
            o_sh = T.alloc_shared((1, block_DV), o_dtype)
            qsq = T.alloc_fragment((1, DK), accum_dtype)
            ksq = T.alloc_fragment((1, DK), accum_dtype)
            ssq = T.alloc_fragment((1,), accum_dtype)
            kS = T.alloc_fragment((block_DV,), accum_dtype)
            oo = T.alloc_fragment((block_DV,), accum_dtype)
            vnew = T.alloc_fragment((block_DV,), accum_dtype)
            decay = T.alloc_fragment((1,), accum_dtype)
            bt = T.alloc_fragment((1,), accum_dtype)

            T.clear(S)
            with T.If(slot >= 0):
                with T.Then():
                    for j_v, j_k in T.Parallel(block_DV, DK):
                        S[j_v, j_k] = pool[slot, bh, v0 + j_v, j_k]

            for t in T.serial(seq_end - seq_start):
                tt = seq_start + t
                T.copy(q[0, tt : tt + 1, bhg, 0:DK], q_s)
                T.copy(k[0, tt : tt + 1, bhg, 0:DK], k_s)
                T.copy(v[0, tt : tt + 1, bh, v0 : v0 + block_DV], v_s)
                # in-kernel qk l2norm: x_n = x / sqrt(sum(x^2) + eps)
                for _i, j_k in T.Parallel(1, DK):
                    qsq[0, j_k] = q_s[0, j_k] * q_s[0, j_k]
                T.reduce_sum(qsq, ssq, dim=1)
                for _i, j_k in T.Parallel(1, DK):
                    q_n[0, j_k] = q_s[0, j_k] * T.rsqrt(ssq[0] + l2norm_eps)
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
                for j_v, j_k in T.Parallel(block_DV, DK):
                    prod[j_v, j_k] = q_n[0, j_k] * S[j_v, j_k]
                T.reduce_sum(prod, oo, dim=1)
                for j_v in T.Parallel(block_DV):
                    o_sh[0, j_v] = oo[j_v] * scale
                T.copy(o_sh, o[0, tt : tt + 1, bh, v0 : v0 + block_DV])
                if store_intermediate:
                    with T.If(slot >= 0):
                        with T.Then():
                            for j_v, j_k in T.Parallel(block_DV, DK):
                                ibuf[cslot, t, bh, v0 + j_v, j_k] = S[j_v, j_k]

            if not disable_state_update:
                with T.If(slot >= 0):
                    with T.Then():
                        for j_v, j_k in T.Parallel(block_DV, DK):
                            pool[slot, bh, v0 + j_v, j_k] = S[j_v, j_k]

    return kernel


def fused_recurrent_gdr_verify_gated_fwd(
    q, k, v, a, b, A_log, dt_bias, pool, state_indices, cu_seqlens,
    intermediate_states_buffer, intermediate_state_indices, o,
    scale=None, disable_state_update=True, allow_neg_eigval=False,
):
    """In-kernel fused-gating verify entry: raw a,b,A_log,dt_bias; computes g/beta + qk-l2norm
    inside the kernel. Graph-safe (all buffers caller-provided)."""
    _, total_tokens, Hg, K = k.shape
    _, _, H, V = v.shape
    N = state_indices.shape[0]
    assert K == V == 128 and H % Hg == 0
    scale = scale or K ** -0.5
    store_intermediate = intermediate_states_buffer is not None

    grid_base = N * H  # bandwidth sweet spot (autotuned, H100): block_DV=64 @ threads=128
    block_DV = 64 if grid_base * 2 >= TARGET_NUM_CTAS else 32

    kern = tilelang_fused_recurrent_gdr_verify_gated(
        H, Hg, K, V, scale,
        accum_dtype="float32", qkva_dtype=q.dtype, ab_dtype=a.dtype, gate_dtype=A_log.dtype,
        pool_dtype=pool.dtype, o_dtype=o.dtype, seqlen_dtype=cu_seqlens.dtype,
        idx_dtype=state_indices.dtype, store_intermediate=store_intermediate,
        disable_state_update=disable_state_update, allow_neg_eigval=allow_neg_eigval,
        block_DV=block_DV, threads=max(128, block_DV * 2),
    )
    kern(q, k, v, a, b, A_log, dt_bias, pool, state_indices, cu_seqlens,
         intermediate_state_indices, o, intermediate_states_buffer)
    return o
