# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]
import torch
import tilelang
import tilelang.language as T

MULTI_PROCESSOR_COUNT = torch.cuda.get_device_properties().multi_processor_count
TARGET_NUM_CTAS = int(MULTI_PROCESSOR_COUNT * 0.7)


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True})
def tilelang_fused_recurrent_gdr_fwd(
    H,
    Hg,
    DK,
    DV,
    scale,
    accum_dtype,
    qkva_dtype,
    g_dtype,
    b_dtype,
    h0_dtype,
    ht_dtype,
    o_dtype,
    seqlen_dtype,
    use_initial_state,
    store_final_state,
    has_seqlens,
    block_DV=128,
    threads=128,
):
    batch_size = T.dynamic("batch_size")
    num_tokens = T.dynamic("num_tokens")  # = q_len (D); D fixed per call
    q_shape = (batch_size, num_tokens, Hg, DK)
    v_shape = (batch_size, num_tokens, H, DV)
    g_shape = (batch_size, num_tokens, H)
    s_shape = (batch_size, H, DK, DV)
    n_vt = (DV + block_DV - 1) // block_DV

    @T.prim_func
    def kernel(
        q: T.Tensor(q_shape, qkva_dtype),
        k: T.Tensor(q_shape, qkva_dtype),
        v: T.Tensor(v_shape, qkva_dtype),
        g: T.Tensor(g_shape, g_dtype),
        b: T.Tensor(g_shape, b_dtype),
        h0: T.Tensor(s_shape, h0_dtype),
        seqlens: T.Tensor([batch_size], seqlen_dtype),
        o: T.Tensor(v_shape, o_dtype),
        ht: T.Tensor(s_shape, ht_dtype),
    ):
        # Gemm-free, memory-bound decode recurrence. One CTA owns (sequence bb, V-head bh,
        # V-column-tile bv). State S is kept [block_DV, DK] (V-major rows) in an fp32 fragment;
        # the two GEMVs are reductions over the last (DK) dim and the rank-1 is a T.Parallel
        # outer product. No tensor-core gemm -> no M-padding, no warp-partition constraint.
        with T.Kernel(n_vt * batch_size * H, threads=threads) as (bbhv,):
            bbh = bbhv // n_vt
            bv = bbhv % n_vt
            bb = bbh // H
            bh = bbh % H
            bhg = bh // (H // Hg)
            v0 = bv * block_DV

            L = T.alloc_var("int32")
            L = seqlens[bb] if has_seqlens else num_tokens

            S = T.alloc_fragment((block_DV, DK), accum_dtype)  # state [V-tile, K], fp32
            prod = T.alloc_fragment((block_DV, DK), accum_dtype)
            q_s = T.alloc_shared((1, DK), qkva_dtype)  # 2-D staging (1-D slice copies fail layout)
            k_s = T.alloc_shared((1, DK), qkva_dtype)
            v_s = T.alloc_shared((1, block_DV), qkva_dtype)
            o_sh = T.alloc_shared((1, block_DV), o_dtype)
            kS = T.alloc_fragment((block_DV,), accum_dtype)
            oo = T.alloc_fragment((block_DV,), accum_dtype)
            vnew = T.alloc_fragment((block_DV,), accum_dtype)
            decay = T.alloc_fragment((1,), accum_dtype)
            bt = T.alloc_fragment((1,), accum_dtype)

            if use_initial_state:
                # h0 is K-major [B,H,DK,DV]; load transposed into S[v, dk] = h0[bb,bh,dk,v0+v]
                for j_v, j_k in T.Parallel(block_DV, DK):
                    S[j_v, j_k] = h0[bb, bh, j_k, v0 + j_v]
            else:
                T.clear(S)

            for t in T.serial(L):
                T.copy(q[bb, t : t + 1, bhg, 0:DK], q_s)
                T.copy(k[bb, t : t + 1, bhg, 0:DK], k_s)
                T.copy(v[bb, t : t + 1, bh, v0 : v0 + block_DV], v_s)
                decay[0] = T.exp2(g[bb, t, bh] * 1.442695)  # raw g, exp2
                bt[0] = b[bb, t, bh]
                for j_v, j_k in T.Parallel(block_DV, DK):
                    S[j_v, j_k] *= decay[0]
                # kS[v] = sum_dk k[dk] * S[v, dk]
                for j_v, j_k in T.Parallel(block_DV, DK):
                    prod[j_v, j_k] = k_s[0, j_k] * S[j_v, j_k]
                T.reduce_sum(prod, kS, dim=1)
                # v_new[v] = beta * (v[v] - kS[v])
                for j_v in T.Parallel(block_DV):
                    vnew[j_v] = bt[0] * (v_s[0, j_v] - kS[j_v])
                # rank-1: S[v, dk] += k[dk] * v_new[v]
                for j_v, j_k in T.Parallel(block_DV, DK):
                    S[j_v, j_k] += k_s[0, j_k] * vnew[j_v]
                # o[v] = scale * sum_dk q[dk] * S[v, dk]  (post-update read)
                for j_v, j_k in T.Parallel(block_DV, DK):
                    prod[j_v, j_k] = q_s[0, j_k] * S[j_v, j_k]
                T.reduce_sum(prod, oo, dim=1)
                for j_v in T.Parallel(block_DV):
                    o_sh[0, j_v] = oo[j_v] * scale
                T.copy(o_sh, o[bb, t : t + 1, bh, v0 : v0 + block_DV])

            if store_final_state:
                # ht is K-major [B,H,DK,DV]; store transposed ht[bb,bh,dk,v0+v] = S[v, dk]
                for j_v, j_k in T.Parallel(block_DV, DK):
                    ht[bb, bh, j_k, v0 + j_v] = S[j_v, j_k]

    return kernel


def fused_recurrent_gdr_fwd(
    q,
    k,
    v,
    g,
    beta,
    scale=None,
    initial_state=None,
    output_final_state=False,
    seqlens=None,
):
    B, Tq, Hg, K = k.shape
    _, _, H, V = v.shape
    assert K == V == 128 and H % Hg == 0
    scale = scale or K ** -0.5

    # Memory-bound: occupancy beats bigger tiles. block_DV=64 (2 V-tiles) at threads=128 is
    # the bandwidth sweet spot (autotuned on H100); fall to 32 (4 V-tiles) for the low-CTA tail.
    grid_base = B * H
    block_DV = 64 if grid_base * 2 >= TARGET_NUM_CTAS else 32

    use_initial_state = initial_state is not None
    if initial_state is None:
        initial_state = torch.empty((B, H, K, V), dtype=torch.float32, device=k.device)
    final_state = torch.empty((B, H, K, V), dtype=torch.float32, device=k.device)
    o = torch.empty_like(v)

    has_seqlens = seqlens is not None
    if seqlens is None:
        seqlens = torch.empty((B,), dtype=torch.int32, device=k.device)
    seqlen_dtype = seqlens.dtype

    kern = tilelang_fused_recurrent_gdr_fwd(
        H,
        Hg,
        K,
        V,
        scale,
        accum_dtype="float32",
        qkva_dtype=q.dtype,
        g_dtype=g.dtype,
        b_dtype=beta.dtype,
        h0_dtype=initial_state.dtype,
        ht_dtype=final_state.dtype,
        o_dtype=o.dtype,
        seqlen_dtype=seqlen_dtype,
        use_initial_state=use_initial_state,
        store_final_state=output_final_state,
        has_seqlens=has_seqlens,
        block_DV=block_DV,
        threads=128,
    )
    kern(q, k, v, g, beta, initial_state, seqlens, o, final_state)
    return o, (final_state if output_final_state else None)
