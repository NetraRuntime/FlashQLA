from typing import Optional, Tuple

import torch
import tilelang
import tilelang.language as T
from tilelang.autotuner import autotune

from fla.ops.utils import prepare_chunk_indices, prepare_chunk_offsets


@tilelang.jit(
    # out_idx=[-3, -2, -1],
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
    },
)
def tilelang_fused_chunk_gdr_fwd(
    H,
    Hg,
    DK,
    DV,
    chunk_size,
    scale,
    accum_dtype,
    qkva_dtype,
    g_dtype,
    b_dtype,
    h0_dtype,
    ht_dtype,
    h_dtype,
    o_dtype,
    seqlen_dtype,
    use_initial_state,
    store_final_state,
    store_h,
    store_o,
    is_varlen,
    block_DV=128,
):
    batch_size = T.dynamic('batch_size')
    num_tokens = T.dynamic('num_tokens')
    num_chunks = T.dynamic('num_chunks')
    block_S = chunk_size

    if is_varlen:
        q_shape = (1, num_tokens, Hg, DK)
        k_shape = (1, num_tokens, Hg, DK)
        v_shape = (1, num_tokens, H, DV)
        o_shape = (1, num_tokens, H, DV)
        a_shape = (1, num_tokens, H, chunk_size)
        g_shape = (1, num_tokens, H)
        b_shape = (1, num_tokens, H)
        h_shape = (1, num_chunks, H, DK, DV)
    else:
        q_shape = (batch_size, num_tokens, Hg, DK)
        k_shape = (batch_size, num_tokens, Hg, DK)
        v_shape = (batch_size, num_tokens, H, DV)
        o_shape = (batch_size, num_tokens, H, DV)
        a_shape = (batch_size, num_tokens, H, chunk_size)
        g_shape = (batch_size, num_tokens, H)
        b_shape = (batch_size, num_tokens, H)
        h_shape = (batch_size, num_chunks, H, DK, DV)
    h0_shape = (batch_size, H, DK, DV)
    ht_shape = (batch_size, H, DK, DV)

    @T.prim_func
    def tilelang_fused_chunk_gdr_fwd_kernel(
        q: T.Tensor(q_shape, dtype=qkva_dtype),
        k: T.Tensor(k_shape, dtype=qkva_dtype),
        v: T.Tensor(v_shape, dtype=qkva_dtype),
        a: T.Tensor(a_shape, dtype=qkva_dtype),
        g: T.Tensor(g_shape, dtype=g_dtype),
        b: T.Tensor(b_shape, dtype=b_dtype),
        h0: T.Tensor(h0_shape, dtype=h0_dtype),
        cu_seqlens: T.Tensor([batch_size + 1], dtype=seqlen_dtype),
        chunk_offsets: T.Tensor([batch_size + 1], dtype=seqlen_dtype),
        o: T.Tensor(o_shape, dtype=o_dtype),
        h: T.Tensor(h_shape, dtype=h_dtype),
        ht: T.Tensor(ht_shape, dtype=ht_dtype),
    ):
        with T.Kernel(T.ceildiv(DV, block_DV), batch_size * H, threads=512) as (bv, bbh):
            bb, bh = bbh // H, bbh % H
            bhg = bh // (H // Hg)

            batch_idx = T.alloc_var("int32")
            seq_start_idx = T.alloc_var("int32")
            seq_end_idx = T.alloc_var("int32")
            seq_split_idx = T.alloc_var("int32")
            chunk_start_idx = T.alloc_var("int32")
            chunk_split_idx = T.alloc_var("int32")

            batch_idx = 0 if is_varlen else bb
            seq_start_idx = cu_seqlens[bb] if is_varlen else 0
            seq_end_idx = cu_seqlens[bb + 1] if is_varlen else num_tokens
            chunk_start_idx = chunk_offsets[bb] if is_varlen else 0

            num_iters = T.ceildiv(seq_end_idx - seq_start_idx, block_S * 2)
            num_unmasked_iters = (seq_end_idx - seq_start_idx) // (block_S * 2)
            num_unmasked_chunks = num_unmasked_iters * 2

            q_shared_0 = T.alloc_shared((block_S, DK), dtype=qkva_dtype)
            k_shared_0 = T.alloc_shared((block_S, DK), dtype=qkva_dtype)
            v_shared_0 = T.alloc_shared((block_S, block_DV), dtype=qkva_dtype)
            a_shared_0 = T.alloc_shared((block_S, block_S), dtype=qkva_dtype)
            g_shared_0 = T.alloc_shared((block_S), dtype=accum_dtype, scope="shared")
            g_exp_shared_0 = T.alloc_shared((block_S), dtype=accum_dtype, scope="shared")
            g_rev_exp_shared_0 = T.alloc_shared((block_S), dtype=accum_dtype, scope="shared")
            b_shared_0 = T.alloc_shared((block_S), dtype=accum_dtype, scope="shared")
            q_shared_1 = T.alloc_shared((block_S, DK), dtype=qkva_dtype)
            k_shared_1 = T.alloc_shared((block_S, DK), dtype=qkva_dtype)
            v_shared_1 = T.alloc_shared((block_S, block_DV), dtype=qkva_dtype)
            a_shared_1 = T.alloc_shared((block_S, block_S), dtype=qkva_dtype)
            g_shared_1 = T.alloc_shared((block_S), dtype=accum_dtype, scope="shared")
            g_exp_shared_1 = T.alloc_shared((block_S), dtype=accum_dtype, scope="shared")
            g_rev_exp_shared_1 = T.alloc_shared((block_S), dtype=accum_dtype, scope="shared")
            b_shared_1 = T.alloc_shared((block_S), dtype=accum_dtype, scope="shared")

            o_shared = T.alloc_shared((block_S, block_DV), dtype=o_dtype)
            h_shared = T.alloc_shared((DK, block_DV), dtype=qkva_dtype)
            vd_shared = T.alloc_shared((block_S, block_DV), dtype=qkva_dtype)
            vn_shared = T.alloc_shared((block_S, block_DV), dtype=qkva_dtype)
            p_shared = T.alloc_shared((block_S, block_S), dtype=qkva_dtype)

            h_fragment = T.alloc_fragment((DK, block_DV), dtype=accum_dtype)
            o_fragment = T.alloc_fragment((block_S, block_DV), dtype=accum_dtype)
            v_fragment = T.alloc_fragment((block_S, block_DV), dtype=accum_dtype)
            p_fragment = T.alloc_fragment((block_S, block_S), dtype=accum_dtype)
            g_fragment = T.alloc_fragment((block_S, block_S), dtype=accum_dtype)
            g_last_local = T.alloc_local((1), dtype=accum_dtype)

            data_is_ready_0 = T.alloc_barrier(arrive_count=96)
            data_is_free_0 = T.alloc_barrier(arrive_count=384)
            g_is_ready_0 = T.alloc_barrier(arrive_count=32)
            data_is_ready_1 = T.alloc_barrier(arrive_count=96)
            data_is_free_1 = T.alloc_barrier(arrive_count=384)
            g_is_ready_1 = T.alloc_barrier(arrive_count=32)

            # TODO: check
            bar_o = T.alloc_barrier(arrive_count=128)
            bar_0 = T.alloc_barrier(arrive_count=288)
            bar_1 = T.alloc_barrier(arrive_count=256)
            bar_2 = T.alloc_barrier(arrive_count=256)
            bar_3 = T.alloc_barrier(arrive_count=256)
            bar_4 = T.alloc_barrier(arrive_count=256)
            bar_5 = T.alloc_barrier(arrive_count=416)

            T.annotate_layout({
                q_shared_0: tilelang.layout.make_swizzled_layout(q_shared_0),
                k_shared_0: tilelang.layout.make_swizzled_layout(k_shared_0),
                v_shared_0: tilelang.layout.make_swizzled_layout(v_shared_0),
                a_shared_0: tilelang.layout.make_swizzled_layout(a_shared_0),
                q_shared_1: tilelang.layout.make_swizzled_layout(q_shared_1),
                k_shared_1: tilelang.layout.make_swizzled_layout(k_shared_1),
                v_shared_1: tilelang.layout.make_swizzled_layout(v_shared_1),
                a_shared_1: tilelang.layout.make_swizzled_layout(a_shared_1),
                o_shared: tilelang.layout.make_swizzled_layout(o_shared),
                h_shared: tilelang.layout.make_swizzled_layout(h_shared),
                vd_shared: tilelang.layout.make_swizzled_layout(vd_shared),
                vn_shared: tilelang.layout.make_swizzled_layout(vn_shared),
                p_shared: tilelang.layout.make_swizzled_layout(p_shared),
            })

            # T.use_swizzle(10)

            tx = T.get_thread_binding()

            PRODUCER_WG_IDX = 3
            CONSUMER_V_WG_IDX = 1
            CONSUMER_S_WG_IDX = 0
            CONSUMER_O_WG_IDX = 2
            PRODUCER_NREG = 48
            CONSUMER_V_NREG = 128
            CONSUMER_S_NREG = 184
            CONSUMER_O_NREG = 152

            if tx >= PRODUCER_WG_IDX * 128 and tx < (PRODUCER_WG_IDX + 1) * 128:
                T.set_max_nreg(PRODUCER_NREG, 0)

                if tx < PRODUCER_WG_IDX * 128 + 32:

                    for i_s in T.serial(num_iters):

                        # [STAGE 0]
                        T.barrier_wait(data_is_free_0, (i_s + 1) % 2)
                        left_0 = seq_start_idx + (i_s * 2 + 0) * block_S
                        right_0 = left_0 + block_S

                        # Load Q
                        T.copy(q[batch_idx, left_0:right_0, bhg, 0:DK], q_shared_0)
                        # Load K
                        T.copy(k[batch_idx, left_0:right_0, bhg, 0:DK], k_shared_0)
                        # Load V
                        T.copy(v[batch_idx, left_0:right_0, bh, bv * block_DV:(bv + 1) * block_DV], v_shared_0)
                        # Load A
                        T.copy(a[batch_idx, left_0:right_0, bh, 0:block_S], a_shared_0)

                        T.barrier_arrive(data_is_ready_0)

                        # [STAGE 1]
                        T.barrier_wait(data_is_free_1, (i_s + 1) % 2)
                        left_1 = seq_start_idx + (i_s * 2 + 1) * block_S
                        right_1 = left_1 + block_S

                        # Load Q
                        T.copy(q[batch_idx, left_1:right_1, bhg, 0:DK], q_shared_1)
                        # Load K
                        T.copy(k[batch_idx, left_1:right_1, bhg, 0:DK], k_shared_1)
                        # Load V
                        T.copy(v[batch_idx, left_1:right_1, bh, bv * block_DV:(bv + 1) * block_DV], v_shared_1)
                        # Load A
                        T.copy(a[batch_idx, left_1:right_1, bh, 0:block_S], a_shared_1)

                        T.barrier_arrive(data_is_ready_1)

                elif tx < PRODUCER_WG_IDX * 128 + 64:

                    for i_s in T.serial(num_iters):

                        # [STAGE 0]
                        T.barrier_wait(data_is_free_0, (i_s + 1) % 2)

                        # Load gamma
                        if seq_start_idx + (i_s * 2 + 1) * block_S <= seq_end_idx:
                            for j_s in T.Parallel(block_S):
                                g_shared_0[j_s] = g[batch_idx, seq_start_idx + (i_s * 2 + 0) * block_S + j_s, bh]
                        else:
                            for j_s in T.Parallel(block_S):
                                if seq_start_idx + (i_s * 2 + 0) * block_S + j_s < seq_end_idx:
                                    g_shared_0[j_s] = g[batch_idx, seq_start_idx + (i_s * 2 + 0) * block_S + j_s, bh]
                                else:
                                    g_shared_0[j_s] = g[batch_idx, seq_end_idx - 1, bh]
                        T.barrier_arrive(g_is_ready_0)
                        # Load beta
                        if seq_start_idx + (i_s * 2 + 1) * block_S <= seq_end_idx:
                            for j_s in T.Parallel(block_S):
                                b_shared_0[j_s] = b[batch_idx, seq_start_idx + (i_s * 2 + 0) * block_S + j_s, bh]
                        else:
                            for j_s in T.Parallel(block_S):
                                if seq_start_idx + (i_s * 2 + 0) * block_S + j_s < seq_end_idx:
                                    b_shared_0[j_s] = b[batch_idx, seq_start_idx + (i_s * 2 + 0) * block_S + j_s, bh]
                                else:
                                    b_shared_0[j_s] = 0

                        T.barrier_arrive(data_is_ready_0)

                        # [STAGE 1]
                        T.barrier_wait(data_is_free_1, (i_s + 1) % 2)

                        # Load gamma
                        if seq_start_idx + (i_s * 2 + 2) * block_S <= seq_end_idx:
                            for j_s in T.Parallel(block_S):
                                g_shared_1[j_s] = g[batch_idx, seq_start_idx + (i_s * 2 + 1) * block_S + j_s, bh]
                        else:
                            for j_s in T.Parallel(block_S):
                                if seq_start_idx + (i_s * 2 + 1) * block_S >= seq_end_idx:
                                    g_shared_1[j_s] = 0
                                elif seq_start_idx + (i_s * 2 + 1) * block_S + j_s < seq_end_idx:
                                    g_shared_1[j_s] = g[batch_idx, seq_start_idx + (i_s * 2 + 1) * block_S + j_s, bh]
                                else:
                                    g_shared_1[j_s] = g[batch_idx, seq_end_idx - 1, bh]
                        T.barrier_arrive(g_is_ready_1)
                        # Load beta
                        if seq_start_idx + (i_s * 2 + 2) * block_S <= seq_end_idx:
                            for j_s in T.Parallel(block_S):
                                b_shared_1[j_s] = b[batch_idx, seq_start_idx + (i_s * 2 + 1) * block_S + j_s, bh]
                        else:
                            for j_s in T.Parallel(block_S):
                                if seq_start_idx + (i_s * 2 + 1) * block_S + j_s < seq_end_idx:
                                    b_shared_1[j_s] = b[batch_idx, seq_start_idx + (i_s * 2 + 1) * block_S + j_s, bh]
                                else:
                                    b_shared_1[j_s] = 0

                        T.barrier_arrive(data_is_ready_1)

                elif tx < PRODUCER_WG_IDX * 128 + 96:

                    for i_s in T.serial(num_iters):

                        # [STAGE 0]
                        T.barrier_wait(g_is_ready_0, (i_s + 0) % 2)
                        # Precompute g, g_last/g
                        for j_s in T.Parallel(block_S):
                            g_exp_shared_0[j_s] = T.exp2(g_shared_0[j_s] * 1.442695)
                            g_rev_exp_shared_0[j_s] = T.if_then_else(
                                seq_start_idx + (i_s * 2 + 0) * block_S + j_s < seq_end_idx,
                                T.exp2((g_shared_0[block_S - 1] - g_shared_0[j_s]) * 1.442695),
                                0.0,
                            )
                        T.barrier_arrive(data_is_ready_0)

                        # [STAGE 1]
                        T.barrier_wait(g_is_ready_1, (i_s + 0) % 2)
                        # Precompute g, g_last/g
                        for j_s in T.Parallel(block_S):
                            g_exp_shared_1[j_s] = T.exp2(g_shared_1[j_s] * 1.442695)
                            g_rev_exp_shared_1[j_s] = T.if_then_else(
                                seq_start_idx + (i_s * 2 + 1) * block_S + j_s < seq_end_idx,
                                T.exp2((g_shared_1[block_S - 1] - g_shared_1[j_s]) * 1.442695),
                                0.0,
                            )
                        T.barrier_arrive(data_is_ready_1)

                elif tx < PRODUCER_WG_IDX * 128 + 128:

                    for i_s in T.serial(num_unmasked_chunks):
                        right = seq_start_idx + i_s * block_S
                        left = right - block_S

                        T.barrier_arrive(bar_0)

                        T.barrier_wait(bar_0, i_s % 2)
                        # Store O
                        if i_s > 0 and store_o:
                            T.copy(o_shared, o[batch_idx, left:right, bh, bv * block_DV:(bv + 1) * block_DV])
                        T.barrier_arrive(bar_5)

                        T.barrier_wait(bar_1, i_s % 2)
                        # Store S
                        if store_h:
                            T.copy(h_shared, h[batch_idx, chunk_start_idx + i_s, bh, 0:DK, bv * block_DV:(bv + 1) * block_DV])


                    if num_unmasked_iters < num_iters:
                        seq_split_idx = seq_start_idx + num_unmasked_chunks * block_S
                        chunk_split_idx = chunk_start_idx + num_unmasked_chunks

                        # [STAGE 0]
                        T.barrier_arrive(bar_0)

                        T.barrier_wait(bar_0, 0)

                        # Store O
                        if num_unmasked_chunks > 0 and store_o:
                            T.copy(o_shared, o[batch_idx, seq_split_idx - block_S:seq_split_idx, bh, bv * block_DV:(bv + 1) * block_DV])
                        T.barrier_arrive(bar_5)

                        T.barrier_wait(bar_1, 0)
                        # Store S
                        if store_h:
                            T.copy(h_shared, h[batch_idx, chunk_split_idx, bh, 0:DK, bv * block_DV:(bv + 1) * block_DV])

                        # [STAGE 1]
                        T.barrier_arrive(bar_0)

                        T.barrier_wait(bar_0, 1)
                        # Store O
                        if store_o:
                            for j_s, j_v in T.Parallel(block_S, block_DV):
                                with T.If(seq_split_idx + j_s < seq_end_idx):
                                    with T.Then():
                                        o[batch_idx, seq_split_idx + j_s, bh, bv * block_DV + j_v] = o_shared[j_s, j_v]
                        T.barrier_arrive(bar_5)

                        T.barrier_wait(bar_1, 1)
                        # Store S
                        if seq_start_idx + (num_unmasked_chunks + 1) * block_S < seq_end_idx and store_h:
                            T.copy(h_shared, h[batch_idx, chunk_split_idx + 1, bh, 0:DK, bv * block_DV:(bv + 1) * block_DV])

                    seq_split_idx = seq_start_idx + (num_iters * 2 - 1) * block_S

                    # Store O
                    T.barrier_wait(bar_o, 0)
                    if store_o:
                        for j_s, j_v in T.Parallel(block_S, block_DV):
                            with T.If(seq_split_idx + j_s < seq_end_idx):
                                with T.Then():
                                    o[batch_idx, seq_split_idx + j_s, bh, bv * block_DV + j_v] = o_shared[j_s, j_v]

            elif tx >= CONSUMER_V_WG_IDX * 128 and tx < (CONSUMER_V_WG_IDX + 1) * 128:
                T.set_max_nreg(CONSUMER_V_NREG, 1)

                # Main Loop
                for i_s in T.serial(num_iters):

                    # [STAGE 0]
                    T.barrier_wait(data_is_ready_0, (i_s + 0) % 2)

                    # [STAGE 0] 1
                    T.barrier_wait(bar_1, 0)
                    # U = K @ S
                    T.gemm_v1(k_shared_0, h_shared, v_fragment, clear_accum=True)
                    T.barrier_arrive(bar_2)

                    # [STAGE 0] 2
                    T.barrier_wait(bar_2, 0)
                    # W = b (V - g * U)
                    for j_s, j_v in T.Parallel(block_S, block_DV):
                        v_fragment[j_s, j_v] = v_shared_0[j_s, j_v] - g_exp_shared_0[j_s] * v_fragment[j_s, j_v]
                    for j_s, j_v in T.Parallel(block_S, block_DV):
                        v_fragment[j_s, j_v] *= b_shared_0[j_s]
                    # S2[1] W
                    T.copy(v_fragment, vn_shared)
                    T.barrier_arrive(bar_3)

                    # [STAGE 0] 3
                    T.barrier_wait(bar_3, 0)
                    # Vd = Ag @ W
                    T.gemm_v1(a_shared_0, vn_shared, v_fragment, clear_accum=True)
                    # S2[2] Vd
                    T.copy(v_fragment, vd_shared)
                    T.barrier_arrive(bar_4)

                    T.barrier_arrive(data_is_free_0)

                    # [STAGE 0] 4
                    T.barrier_wait(bar_4, 0)
                    # V' = g_last/g Vd
                    for j_s, j_v in T.Parallel(block_S, block_DV):
                        v_fragment[j_s, j_v] *= g_rev_exp_shared_0[j_s]
                    # S2[1] V'
                    T.copy(v_fragment, vn_shared)
                    T.barrier_arrive(bar_5)

                    # [STAGE 1]
                    T.barrier_wait(data_is_ready_1, (i_s + 0) % 2)

                    # [STAGE 1] 1
                    T.barrier_wait(bar_1, 1)
                    # U = K @ S
                    T.gemm_v1(k_shared_1, h_shared, v_fragment, clear_accum=True)
                    T.barrier_arrive(bar_2)

                    # [STAGE 1] 2
                    T.barrier_wait(bar_2, 1)
                    # W = b (V - g * U)
                    for j_s, j_v in T.Parallel(block_S, block_DV):
                        v_fragment[j_s, j_v] = v_shared_1[j_s, j_v] - g_exp_shared_1[j_s] * v_fragment[j_s, j_v]
                    for j_s, j_v in T.Parallel(block_S, block_DV):
                        v_fragment[j_s, j_v] *= b_shared_1[j_s]
                    # S2[1] W
                    T.copy(v_fragment, vn_shared)
                    T.barrier_arrive(bar_3)

                    # [STAGE 1] 3
                    T.barrier_wait(bar_3, 1)
                    # Vd = Ag @ W
                    T.gemm_v1(a_shared_1, vn_shared, v_fragment, clear_accum=True)
                    # S2[2] Vd
                    T.copy(v_fragment, vd_shared)
                    T.barrier_arrive(bar_4)

                    T.barrier_arrive(data_is_free_1)

                    # [STAGE 1] 4
                    T.barrier_wait(bar_4, 1)
                    # V' = g_last/g Vd
                    for j_s, j_v in T.Parallel(block_S, block_DV):
                        v_fragment[j_s, j_v] *= g_rev_exp_shared_1[j_s]
                    # S2[1] V'
                    T.copy(v_fragment, vn_shared)
                    T.barrier_arrive(bar_5)

            elif tx >= CONSUMER_S_WG_IDX * 128 and tx < (CONSUMER_S_WG_IDX + 1) * 128:
                T.set_max_nreg(CONSUMER_S_NREG, 1)

                # Initialize S
                if use_initial_state:
                    T.copy(h0[bb, bh, 0:DK, bv * block_DV:(bv + 1) * block_DV], h_fragment)
                else:
                    T.clear(h_fragment)

                # Main Loop
                for i_s in T.serial(num_iters):

                    # [STAGE 0]
                    T.barrier_wait(data_is_ready_0, (i_s + 0) % 2)
                    T.barrier_arrive(bar_0)

                    # [STAGE 0] 0
                    T.barrier_wait(bar_0, 0)
                    # S4[S] S
                    T.copy(h_fragment, h_shared)
                    T.barrier_arrive(bar_1)

                    # [STAGE 0] 2, 3, 4
                    T.barrier_wait(bar_2, 0)
                    # S = g_last * S
                    g_last_local[0] = g_exp_shared_0[block_S - 1]
                    for j_k, j_v in T.Parallel(DK, block_DV):
                        h_fragment[j_k, j_v] *= g_last_local[0]
                    T.barrier_arrive(bar_5)

                    # [STAGE 0] 5
                    T.barrier_wait(bar_5, 0)
                    # S += K^T @ V'
                    T.gemm_v1(k_shared_0, vn_shared, h_fragment, transpose_A=True, clear_accum=False)

                    T.barrier_arrive(data_is_free_0)

                    # [STAGE 1]
                    T.barrier_wait(data_is_ready_1, (i_s + 0) % 2)
                    T.barrier_arrive(bar_0)

                    # [STAGE 1] 0
                    T.barrier_wait(bar_0, 1)
                    # S4[S] S
                    T.copy(h_fragment, h_shared)
                    T.barrier_arrive(bar_1)

                    # [STAGE 1] 2, 3, 4
                    T.barrier_wait(bar_2, 1)
                    # S = g_last * S
                    g_last_local[0] = g_exp_shared_1[block_S - 1]
                    for j_k, j_v in T.Parallel(DK, block_DV):
                        h_fragment[j_k, j_v] *= g_last_local[0]
                    T.barrier_arrive(bar_5)

                    # [STAGE 1] 5
                    T.barrier_wait(bar_5, 1)
                    # S += K^T @ V'
                    T.gemm_v1(k_shared_1, vn_shared, h_fragment, transpose_A=True, clear_accum=False)

                    T.barrier_arrive(data_is_free_1)

                # Store final S
                if store_final_state:
                    T.copy(h_fragment, ht[bb, bh, 0:DK, bv * block_DV:(bv + 1) * block_DV])

            elif tx >= CONSUMER_O_WG_IDX * 128 and tx < (CONSUMER_O_WG_IDX + 1) * 128:
                T.set_max_nreg(CONSUMER_O_NREG, 1)

                # Main Loop
                for i_s in T.serial(num_iters):

                    # [STAGE 0]
                    T.barrier_wait(data_is_ready_0, (i_s + 0) % 2)
                    T.barrier_arrive(bar_0)

                    # [STAGE 0] 0
                    T.barrier_wait(bar_0, 0)
                    # P = Q K^T
                    T.gemm_v1(q_shared_0, k_shared_0, p_fragment, transpose_B=True, clear_accum=True)
                    T.barrier_arrive(bar_1)

                    # [STAGE 0] 1
                    T.barrier_wait(bar_1, 0)
                    # G = Lower(diag(g) @ I @ diag(1/g))
                    for j_s, j_t in T.Parallel(block_S, block_S):
                        with T.If(j_s >= j_t):
                            with T.Then():
                                g_fragment[j_s, j_t] = T.exp2((g_shared_0[j_s] - g_shared_0[j_t]) * 1.442695)
                            with T.Else():
                                g_fragment[j_s, j_t] = 0
                    # Pg = s * G * P
                    for j_s, j_t in T.Parallel(block_S, block_S):
                        p_fragment[j_s, j_t] *= scale * g_fragment[j_s, j_t]
                    # S1[1] Pg
                    T.copy(p_fragment, p_shared)
                    # Ag = G * Ar
                    T.copy(a_shared_0, p_fragment)
                    for j_s, j_t in T.Parallel(block_S, block_S):
                        p_fragment[j_s, j_t] *= g_fragment[j_s, j_t]
                    T.copy(p_fragment, a_shared_0)
                    T.barrier_arrive(bar_2)

                    # [STAGE 0] 2
                    T.barrier_wait(bar_2, 0)
                    # O = Q @ S
                    T.gemm_v1(q_shared_0, h_shared, o_fragment, clear_accum=True)
                    T.barrier_arrive(bar_3)

                    T.barrier_arrive(data_is_free_0)

                    # [STAGE 0] 3
                    T.barrier_wait(bar_3, 0)
                    # O = s * g * O
                    for j_s, j_k in T.Parallel(block_S, DK):
                        o_fragment[j_s, j_k] *= scale * g_exp_shared_0[j_s]
                    T.barrier_arrive(bar_4)

                    # [STAGE 0] 4
                    T.barrier_wait(bar_4, 0)
                    # O += Pg @ Vd
                    T.gemm_v1(p_shared, vd_shared, o_fragment, clear_accum=False)
                    T.barrier_arrive(bar_5)

                    # [STAGE 0] 5
                    T.barrier_wait(bar_5, 0)
                    T.copy(o_fragment, o_shared)

                    # [STAGE 1]
                    T.barrier_wait(data_is_ready_1, (i_s + 0) % 2)
                    T.barrier_arrive(bar_0)

                    # [STAGE 1] 0
                    T.barrier_wait(bar_0, 1)
                    # P = Q K^T
                    T.gemm_v1(q_shared_1, k_shared_1, p_fragment, transpose_B=True, clear_accum=True)
                    T.barrier_arrive(bar_1)

                    # [STAGE 1] 1
                    T.barrier_wait(bar_1, 1)
                    # G = Lower(diag(g) @ I @ diag(1/g))
                    for j_s, j_t in T.Parallel(block_S, block_S):
                        with T.If(j_s >= j_t):
                            with T.Then():
                                g_fragment[j_s, j_t] = T.exp2((g_shared_1[j_s] - g_shared_1[j_t]) * 1.442695)
                            with T.Else():
                                g_fragment[j_s, j_t] = 0
                    # Pg = s * G * P
                    for j_s, j_t in T.Parallel(block_S, block_S):
                        p_fragment[j_s, j_t] *= scale * g_fragment[j_s, j_t]
                    # S1[1] Pg
                    T.copy(p_fragment, p_shared)
                    # Ag = G * Ar
                    T.copy(a_shared_1, p_fragment)
                    for j_s, j_t in T.Parallel(block_S, block_S):
                        p_fragment[j_s, j_t] *= g_fragment[j_s, j_t]
                    T.copy(p_fragment, a_shared_1)
                    T.barrier_arrive(bar_2)

                    # [STAGE 1] 2
                    T.barrier_wait(bar_2, 1)
                    # O = Q @ S
                    T.gemm_v1(q_shared_1, h_shared, o_fragment, clear_accum=True)
                    T.barrier_arrive(bar_3)

                    T.barrier_arrive(data_is_free_1)

                    # [STAGE 1] 3
                    T.barrier_wait(bar_3, 1)
                    # O = s * g * O
                    for j_s, j_k in T.Parallel(block_S, DK):
                        o_fragment[j_s, j_k] *= scale * g_exp_shared_1[j_s]
                    T.barrier_arrive(bar_4)

                    # [STAGE 1] 4
                    T.barrier_wait(bar_4, 1)
                    # O += Pg @ Vd
                    T.gemm_v1(p_shared, vd_shared, o_fragment, clear_accum=False)
                    T.barrier_arrive(bar_5)

                    # [STAGE 1] 5
                    T.barrier_wait(bar_5, 1)
                    T.copy(o_fragment, o_shared)
                
                T.barrier_arrive(bar_o)

    return tilelang_fused_chunk_gdr_fwd_kernel


def fused_gdr_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    scale: float = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = True,
    output_h: bool = True,
    output_o: bool = True,
    chunk_size: int = 64,
    cu_seqlens: Optional[torch.LongTensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size, num_tokens, Hg, K = k.shape
    _, _, H, V = v.shape
    scale = scale or K ** (-0.5)
    assert K == V == 128

    if cu_seqlens is None:
        real_batch_size = batch_size
        num_chunks = tilelang.cdiv(num_tokens, chunk_size)
        cu_seqlens = torch.empty((batch_size + 1), dtype=torch.int32, device=k.device)
        chunk_offsets = torch.empty((batch_size + 1), dtype=torch.int32, device=k.device)
        is_varlen = False
    else:
        real_batch_size = len(cu_seqlens) - 1
        num_chunks = len(prepare_chunk_indices(cu_seqlens, chunk_size))
        chunk_offsets = prepare_chunk_offsets(cu_seqlens, chunk_size).to(cu_seqlens.dtype)
        is_varlen = True

    use_initial_state = initial_state is not None
    if initial_state is None:
        initial_state = torch.empty((real_batch_size, H, K, V), dtype=torch.float32, device=k.device)
    h = torch.empty((batch_size, num_chunks, H, K, V), dtype=k.dtype, device=k.device)
    final_state = torch.empty((real_batch_size, H, K, V), dtype=torch.float32, device=k.device)
    o = torch.empty_like(v)

    if real_batch_size * H <= 32:
        block_DV = 32
    elif real_batch_size * H <= 64:
        block_DV = 64
    else:
        block_DV = 128

    dtypes = {
        'qkva_dtype': str(q.dtype).split('.')[-1],
        'g_dtype': str(g.dtype).split('.')[-1],
        'b_dtype': str(b.dtype).split('.')[-1],
        'h0_dtype': str(initial_state.dtype).split('.')[-1],
        'ht_dtype': str(final_state.dtype).split('.')[-1],
        'h_dtype': str(h.dtype).split('.')[-1],
        'o_dtype': str(o.dtype).split('.')[-1],
        'seqlen_dtype': str(cu_seqlens.dtype).split('.')[-1],
        'accum_dtype': 'float32',
    }
    tilelang_fused_chunk_gdr_fwd_kernel = tilelang_fused_chunk_gdr_fwd(
        H, Hg, K, V, chunk_size, scale,
        use_initial_state=use_initial_state, store_final_state=output_final_state,
        store_h=output_h, store_o=output_o, is_varlen=is_varlen,
        block_DV=block_DV, **dtypes,
    )
    tilelang_fused_chunk_gdr_fwd_kernel(
        q, k, v, a, g, b, initial_state,
        cu_seqlens, chunk_offsets,
        o, h, final_state,
    )

    if not output_final_state:
        final_state = None
    if not output_h:
        h = None
    if not output_o:
        o = None

    return o, h, final_state
