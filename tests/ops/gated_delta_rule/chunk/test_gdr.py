from typing import Optional, Tuple

import torch
import pandas as pd

from fla.ops.gated_delta_rule.compress_heads import compress_heads
# from fla.ops.gated_delta_rule.chunk import chunk_gated_delta_rule as chunk_gated_delta_rule_old
from fla.ops.gated_delta_rule.wy_fast import recompute_w_u_fwd, prepare_wy_repr_bwd
from fla.ops.common.chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd
from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_fwd_h, chunk_gated_delta_rule_bwd_dhu
from fla.ops.common.chunk_o import chunk_fwd_o, chunk_bwd_dv_local, chunk_bwd_dqkwg
from fla.ops.utils import chunk_local_cumsum, solve_tril, prepare_chunk_indices, prepare_chunk_offsets, prepare_lens

# from flav2.ops import chunk_gated_delta_rule as chunk_gated_delta_rule_new
from flav2.ops.gated_delta_rule.chunk import fused_gdr_fwd, fused_gdr_bwd
from flav2.utils import l2norm, pack, unpack, profile, TIMING_LOGGER

from ref_gdr import chunk_gated_delta_rule_fwd as chunk_gated_delta_rule_fwd_ref
from ref_gdr import chunk_gated_delta_rule_bwd as chunk_gated_delta_rule_bwd_ref


def chunk_gated_delta_rule_fwd_old(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor = None,
    cu_seqlens: Optional[torch.LongTensor] = None
):
    g = chunk_local_cumsum(g, chunk_size=64, cu_seqlens=cu_seqlens)

    TIMING_LOGGER('[fwd] kkt')
    A = chunk_scaled_dot_kkt_fwd(
        k=k,
        g=g,
        beta=beta,
        cu_seqlens=cu_seqlens,
        output_dtype=torch.float32
    )
    TIMING_LOGGER('[fwd] kkt')

    TIMING_LOGGER('[fwd] solve')
    A = solve_tril(
        A=A,
        cu_seqlens=cu_seqlens,
        output_dtype=k.dtype
    )
    TIMING_LOGGER('[fwd] solve')

    TIMING_LOGGER('[fwd] wu')
    w, u = recompute_w_u_fwd(
        k=k,
        v=v,
        beta=beta,
        A=A,
        g=g,
        cu_seqlens=cu_seqlens,
    )
    TIMING_LOGGER('[fwd] wu')

    TIMING_LOGGER('[fwd] gdr')
    h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
        k=k,
        w=w,
        u=u,
        g=g,
        initial_state=initial_state,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
    )
    TIMING_LOGGER('[fwd] gdr')

    TIMING_LOGGER('[fwd] o')
    o = chunk_fwd_o(
        q=q,
        k=k,
        v=v_new,
        h=h,
        g=g,
        scale=scale,
        cu_seqlens=cu_seqlens,
    )
    TIMING_LOGGER('[fwd] o')

    return o, h, final_state


def chunk_gated_delta_rule_fwd_new(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
    output_final_state: bool = True,
    output_h: bool = True,
):
    g = chunk_local_cumsum(g, chunk_size=64, cu_seqlens=cu_seqlens)

    TIMING_LOGGER('[fwd] kkt')
    A = chunk_scaled_dot_kkt_fwd(
        k=k,
        g=None,
        beta=beta,
        cu_seqlens=cu_seqlens,
        output_dtype=torch.float32
    )
    TIMING_LOGGER('[fwd] kkt')

    TIMING_LOGGER('[fwd] solve')
    A = solve_tril(
        A=A,
        cu_seqlens=cu_seqlens,
        output_dtype=k.dtype
    )
    TIMING_LOGGER('[fwd] solve')

    TIMING_LOGGER('[fwd] gdr')
    o, h, final_state = fused_gdr_fwd(
        q=q,
        k=k,
        v=v,
        a=A,
        g=g,
        b=beta,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        output_h=output_h,
        output_o=True,
        cu_seqlens=cu_seqlens,
    )
    TIMING_LOGGER('[fwd] gdr')

    return o, h, final_state


def chunk_gated_delta_rule_bwd_old(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    do: torch.Tensor,
    dht: torch.Tensor,
    cu_seqlens: Optional[torch.LongTensor] = None,
):
    g = chunk_local_cumsum(g, chunk_size=64, cu_seqlens=cu_seqlens)

    TIMING_LOGGER('[bwd] rc')
    w, u = recompute_w_u_fwd(
        k=k,
        v=v,
        beta=beta,
        A=A,
        g=g,
        cu_seqlens=cu_seqlens,
    )

    h, v_new, _ = chunk_gated_delta_rule_fwd_h(
        k=k,
        w=w,
        u=u,
        g=g,
        initial_state=initial_state,
        output_final_state=False,
        cu_seqlens=cu_seqlens,
    )
    TIMING_LOGGER('[bwd] rc')

    TIMING_LOGGER('[bwd] dv')
    dv = chunk_bwd_dv_local(
        q=q,
        k=k,
        g=g,
        do=do,
        scale=scale,
        cu_seqlens=cu_seqlens,
    )
    TIMING_LOGGER('[bwd] dv')

    TIMING_LOGGER('[bwd] gdr')
    dh, dh0, dv = chunk_gated_delta_rule_bwd_dhu(
        q=q,
        k=k,
        w=w,
        g=g,
        h0=initial_state,
        dht=dht,
        do=do,
        dv=dv,
        scale=scale,
        cu_seqlens=cu_seqlens,
    )
    TIMING_LOGGER('[bwd] gdr')

    TIMING_LOGGER('[bwd] dqkwg')
    dq, dk1, dw, dg1 = chunk_bwd_dqkwg(
        q=q,
        k=k,
        v=v_new,
        w=w,
        g=g,
        h=h,
        dv=dv,
        do=do,
        dh=dh,
        scale=scale,
        cu_seqlens=cu_seqlens,
    )
    TIMING_LOGGER('[bwd] dqkwg')

    TIMING_LOGGER('[bwd] wy')
    dk, dv, db, dg = prepare_wy_repr_bwd(
        k=k,
        v=v,
        beta=beta,
        g=g,
        A=A,
        dw=dw,
        du=dv,
        dk1=dk1,
        dg1=dg1,
        cu_seqlens=cu_seqlens,
    )
    TIMING_LOGGER('[bwd] wy')

    TIMING_LOGGER('[bwd] cmprs')
    Hg, H = k.shape[-2], v.shape[-2]
    if Hg < H:
        dq, dk = compress_heads(dq, dk, k)
    assert dg.dtype == torch.float32, "dg should be fp32"
    TIMING_LOGGER('[bwd] cmprs')

    dg = chunk_local_cumsum(dg, chunk_size=64, reverse=True, cu_seqlens=cu_seqlens)
    return dq, dk, dv, db, dg, dh0


def chunk_gated_delta_rule_bwd_new(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    do: torch.Tensor,
    dht: torch.Tensor,
    cu_seqlens: Optional[torch.LongTensor] = None,
):
    g = chunk_local_cumsum(g, chunk_size=64, cu_seqlens=cu_seqlens)

    TIMING_LOGGER('[bwd] rc')
    _, h, _ = fused_gdr_fwd(
        q=q,
        k=k,
        v=v,
        a=A,
        g=g,
        b=beta,
        scale=scale,
        initial_state=initial_state,
        output_final_state=False,
        output_h=True,
        output_o=False,
        cu_seqlens=cu_seqlens,
    )
    TIMING_LOGGER('[bwd] rc')

    TIMING_LOGGER('[bwd] gdr')
    dq, dk, dv, dg, db, dh0 = fused_gdr_bwd(
        q=q,
        k=k,
        v=v,
        a=A,
        g=g,
        b=beta,
        do=do,
        dht=dht,
        h=h,
        scale=scale,
        cu_seqlens=cu_seqlens,
    )
    TIMING_LOGGER('[bwd] gdr')

    TIMING_LOGGER('[bwd] cmprs')
    Hg, H = k.shape[-2], v.shape[-2]
    if Hg < H:
        dq, dk = compress_heads(dq, dk, k)
    assert dg.dtype == torch.float32, "dg should be fp32"
    TIMING_LOGGER('[bwd] cmprs')

    dg = chunk_local_cumsum(dg, chunk_size=64, reverse=True, cu_seqlens=cu_seqlens)
    return dq, dk, dv, db, dg, dh0


def test_gated_delta_rule(
    batch_size: int,
    num_tokens: int,
    num_k_heads: int,
    num_v_heads: int,
    head_dim_k: int,
    head_dim_v: int,
    varlen: bool = False,
    chunk_size: int = 64,
    data_dtype: torch.dtype = torch.bfloat16,
    ref_dtype: torch.dtype = torch.float64,
    device: torch.device = 'cuda',
    random_seed: int = 42,
):
    torch.manual_seed(random_seed)
    q = l2norm(torch.randn((batch_size, num_tokens, num_k_heads, head_dim_k), device=device, dtype=data_dtype))
    k = l2norm(torch.randn((batch_size, num_tokens, num_k_heads, head_dim_k), device=device, dtype=data_dtype))
    v = torch.randn((batch_size, num_tokens, num_v_heads, head_dim_v), device=device, dtype=data_dtype)
    g = torch.nn.functional.logsigmoid(torch.randn((batch_size, num_tokens, num_v_heads), device=device, dtype=torch.float32)) / 16
    beta = torch.randn((batch_size, num_tokens, num_v_heads), device=device, dtype=torch.float32).sigmoid()
    h0 = torch.randn((batch_size, num_v_heads, head_dim_k, head_dim_v), device=device, dtype=torch.float32)
    do = torch.randn_like(v)
    dht = torch.randn((batch_size, num_v_heads, head_dim_k, head_dim_v), device=device, dtype=torch.float32) / 8
    scale = head_dim_k ** (-0.5)
    print(f'Shape: B={batch_size} Hk={num_k_heads} Hv={num_v_heads} T={num_tokens} VarLen={varlen}')

    if varlen:
        cu_seqlens = torch.randint(1, num_tokens, (batch_size, ), device=device, dtype=torch.int32)
        cu_seqlens = torch.nn.functional.pad(torch.cumsum(cu_seqlens, dim=-1), (1, 0))
        q = pack(q, cu_seqlens)
        k = pack(k, cu_seqlens)
        v = pack(v, cu_seqlens)
        g = pack(g, cu_seqlens)
        beta = pack(beta, cu_seqlens)
        do = pack(do, cu_seqlens)
    else:
        cu_seqlens = None

    o_old, h_old, s_old = chunk_gated_delta_rule_fwd_old(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=h0,
        cu_seqlens=cu_seqlens,
    )
    o_new, h_new, s_new = chunk_gated_delta_rule_fwd_new(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=h0,
        cu_seqlens=cu_seqlens,
        output_final_state=True,
        output_h=True,
    )
    o_ref, h_ref, s_ref = chunk_gated_delta_rule_fwd_ref(
        q=q.to(ref_dtype, copy=True),
        k=k.to(ref_dtype, copy=True),
        v=v.to(ref_dtype, copy=True),
        g=g.to(ref_dtype, copy=True),
        beta=beta.to(ref_dtype, copy=True),
        scale=scale,
        initial_state=h0,
        cu_seqlens=cu_seqlens,
    )

    print(f'h_old: {(h_old - h_ref).abs().max().item():.4f} / {h_ref.abs().max().item():.4f}')
    print(f'h_new: {(h_new - h_ref).abs().max().item():.4f} / {h_ref.abs().max().item():.4f}')
    print(f's_old: {(s_old - s_ref).abs().max().item():.4f} / {s_ref.abs().max().item():.4f}')
    print(f's_new: {(s_new - s_ref).abs().max().item():.4f} / {s_ref.abs().max().item():.4f}')
    print(f'o_old: {(o_old - o_ref).abs().max().item():.4f} / {o_ref.abs().max().item():.4f}')
    print(f'o_new: {(o_new - o_ref).abs().max().item():.4f} / {o_ref.abs().max().item():.4f}')

    results = {
        'old': profile(chunk_gated_delta_rule_fwd_old, [q, k, v, g, beta, scale, h0, cu_seqlens]),
        'new': profile(chunk_gated_delta_rule_fwd_new, [q, k, v, g, beta, scale, h0, cu_seqlens, True, False]),
    }
    df = pd.DataFrame(results)
    print(df.round(3))
    speedup = results['old']['total'] / results['new']['total']
    print(f'Speed up: {(speedup - 1) * 100:2.2f}%')
    # import ipdb; ipdb.set_trace()
    # with open('output/tmp-speedup.txt', 'a') as f:
    #     f.write(f'{(speedup - 1) * 100:2.2f}\n')
    # return

    A_old = solve_tril(
        A=chunk_scaled_dot_kkt_fwd(
            k=k,
            g=chunk_local_cumsum(g, chunk_size=64, cu_seqlens=cu_seqlens),
            beta=beta,
            cu_seqlens=cu_seqlens,
            output_dtype=torch.float32,
        ),
        cu_seqlens=cu_seqlens,
        output_dtype=k.dtype,
    )
    dq_ref, dk_ref, dv_ref, db_ref, dg_ref, dh0_ref = chunk_gated_delta_rule_bwd_ref(
        q.to(ref_dtype, copy=True),
        k.to(ref_dtype, copy=True),
        v.to(ref_dtype, copy=True),
        g.to(ref_dtype, copy=True),
        beta.to(ref_dtype, copy=True),
        A_old.to(ref_dtype, copy=True),
        scale,
        h0,
        do.to(ref_dtype, copy=True),
        dht.to(ref_dtype, copy=True),
        cu_seqlens,
    )
    dq_old, dk_old, dv_old, db_old, dg_old, dh0_old = chunk_gated_delta_rule_bwd_old(
        q, k, v, g, beta, A_old, scale, h0, do, dht, cu_seqlens,
    )
    A_new = solve_tril(
        A=chunk_scaled_dot_kkt_fwd(k=k, g=None, beta=beta, cu_seqlens=cu_seqlens, output_dtype=torch.float32),
        cu_seqlens=cu_seqlens,
        output_dtype=k.dtype,
    )
    dq_new, dk_new, dv_new, db_new, dg_new, dh0_new = chunk_gated_delta_rule_bwd_new(
        q, k, v, g, beta, A_new, scale, h0, do, dht, cu_seqlens,
    )

    print(f'dq_old: {(dq_old - dq_ref).abs().max().item():.4f} / {dq_ref.abs().max().item():.4f}')
    print(f'dq_new: {(dq_new - dq_ref).abs().max().item():.4f} / {dq_ref.abs().max().item():.4f}')
    print(f'dk_old: {(dk_old - dk_ref).abs().max().item():.4f} / {dk_ref.abs().max().item():.4f}')
    print(f'dk_new: {(dk_new - dk_ref).abs().max().item():.4f} / {dk_ref.abs().max().item():.4f}')
    print(f'dv_old: {(dv_old - dv_ref).abs().max().item():.4f} / {dv_ref.abs().max().item():.4f}')
    print(f'dv_new: {(dv_new - dv_ref).abs().max().item():.4f} / {dv_ref.abs().max().item():.4f}')
    print(f'dh0_old: {(dh0_old - dh0_ref).abs().max().item():.4f} / {dh0_ref.abs().max().item():.4f}')
    print(f'dh0_new: {(dh0_new - dh0_ref).abs().max().item():.4f} / {dh0_ref.abs().max().item():.4f}')
    print(f'db_old: {(db_old - db_ref).abs().max().item():.4f} / {db_ref.abs().max().item():.4f}')
    print(f'db_new: {(db_new - db_ref).abs().max().item():.4f} / {db_ref.abs().max().item():.4f}')
    print(f'dg_old: {(dg_old - dg_ref).abs().max().item():.4f} / {dg_ref.abs().max().item():.4f}')
    print(f'dg_new: {(dg_new - dg_ref).abs().max().item():.4f} / {dg_ref.abs().max().item():.4f}')
    # import ipdb; ipdb.set_trace()

    results = {
        'old': profile(chunk_gated_delta_rule_bwd_old, [q, k, v, g, beta, A_old, scale, h0, do, dht, cu_seqlens]),
        'new': profile(chunk_gated_delta_rule_bwd_new, [q, k, v, g, beta, A_new, scale, h0, do, dht, cu_seqlens]),
    }
    df = pd.DataFrame(results)
    print(df.round(3))
    speedup = results['old']['total'] / results['new']['total']
    print(f'Speed up: {(speedup - 1) * 100:2.2f}%')


if __name__ == '__main__':
    # TODO: check old accuracy
    test_gated_delta_rule(
        batch_size=1,
        num_tokens=32768,
        num_k_heads=16,
        num_v_heads=128,
        head_dim_k=128,
        head_dim_v=128,
        varlen=False,
    )
    exit()
    for num_v_heads in [32, 64, 128]:
        print('-' * 64)
        test_gated_delta_rule(
            batch_size=11,
            num_tokens=33,
            num_k_heads=16,
            num_v_heads=num_v_heads,
            head_dim_k=128,
            head_dim_v=128,
            varlen=False,
        )
        print('-' * 64)
        test_gated_delta_rule(
            batch_size=7,
            num_tokens=4321,
            num_k_heads=16,
            num_v_heads=num_v_heads,
            head_dim_k=128,
            head_dim_v=128,
            varlen=False,
        )
        print('-' * 64)
        test_gated_delta_rule(
            batch_size=3,
            num_tokens=16789,
            num_k_heads=16,
            num_v_heads=num_v_heads,
            head_dim_k=128,
            head_dim_v=128,
            varlen=True,
        )
        print('-' * 64)
        test_gated_delta_rule(
            batch_size=5,
            num_tokens=8192,
            num_k_heads=16,
            num_v_heads=num_v_heads,
            head_dim_k=128,
            head_dim_v=128,
            varlen=True,
        )
        print('-' * 64)
        test_gated_delta_rule(
            batch_size=1,
            num_tokens=4096,
            num_k_heads=16,
            num_v_heads=num_v_heads,
            head_dim_k=128,
            head_dim_v=128,
            varlen=False,
        )
        print('-' * 64)
        test_gated_delta_rule(
            batch_size=1,
            num_tokens=8192,
            num_k_heads=16,
            num_v_heads=num_v_heads,
            head_dim_k=128,
            head_dim_v=128,
            varlen=False,
        )
        print('-' * 64)
        test_gated_delta_rule(
            batch_size=1,
            num_tokens=16384,
            num_k_heads=16,
            num_v_heads=num_v_heads,
            head_dim_k=128,
            head_dim_v=128,
            varlen=False,
        )
        print('-' * 64)
        test_gated_delta_rule(
            batch_size=1,
            num_tokens=32768,
            num_k_heads=16,
            num_v_heads=num_v_heads,
            head_dim_k=128,
            head_dim_v=128,
            varlen=False,
        )
        print('-' * 64)
