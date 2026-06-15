# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]
import torch
import tilelang

from flash_qla.utils import l2norm

if tilelang.contrib.nvcc.get_target_compute_version() == "9.0":
    from .hopper import fused_recurrent_gdr_fwd  # noqa: F401
else:
    raise ValueError("FlashQLA now support sm90 only.")

__all__ = ["fused_recurrent_gdr_fwd", "recurrent_gated_delta_rule"]


def recurrent_gated_delta_rule(
    q,
    k,
    v,
    g,
    beta,
    scale=None,
    initial_state=None,
    output_final_state=False,
    use_qk_l2norm_in_kernel=False,
    seqlens=None,
    head_first=False,
):
    assert q.dtype == k.dtype == v.dtype and q.dtype != torch.float32
    assert not head_first, "head_first=True is not supported."
    assert v.shape[2] % k.shape[2] == 0 and q.shape[-1] == v.shape[-1] == 128
    if scale is None:
        scale = k.shape[-1] ** -0.5
    if use_qk_l2norm_in_kernel:
        q = l2norm(q)
        k = l2norm(k)
    o, final_state = fused_recurrent_gdr_fwd(
        q,
        k,
        v,
        g,
        beta,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        seqlens=seqlens,
    )
    return o.to(q.dtype), final_state
