# tests/test_replay_gdr.py
"""Parity tests for the DFlash K0 reduced-cache REPLAY kernel.

The replay kernel must produce the SAME committed GDN state as the verify kernel, because
in K0 the accepted-tail state is recomputed from the pre-verify pool state. Two checks:

1. FULL-tail parity: replay over the whole block (tail=D) from the pre-verify state ==
   gated-verify-with-commit final state. Same recurrence body, same tokens, same initial
   state -> expect bit-identical (the q-projection/output that verify also computes does
   not touch the state S).

2. PARTIAL-tail parity: replay over tail=L == the verify kernel's own intermediate state
   after L tokens (ibuf[:, L-1]). This is exactly the K0 contract: accept L tokens ->
   commit the state the target passed through at step L.
"""
import pytest
import torch

from flash_qla.ops.gated_delta_rule.fused_recurrent.hopper.fused_recurrent_verify import (
    fused_recurrent_gdr_verify_gated_fwd,
)
from flash_qla.ops.gated_delta_rule.fused_recurrent.hopper.fused_recurrent_replay import (
    fused_recurrent_gdr_replay_fwd,
)

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _mk(N, D, Hk, Hv, num_slots, seed=0):
    """Full-block layout (every request occupies D tokens back-to-back) + raw gate inputs."""
    torch.manual_seed(seed)
    total = N * D
    dev, bf16 = "cuda", torch.bfloat16
    q = torch.randn(1, total, Hk, 128, device=dev, dtype=bf16)
    k = torch.randn(1, total, Hk, 128, device=dev, dtype=bf16)
    v = torch.randn(1, total, Hv, 128, device=dev, dtype=bf16)
    a = torch.randn(1, total, Hv, device=dev, dtype=torch.float32)
    b = torch.randn(1, total, Hv, device=dev, dtype=torch.float32)
    A_log = torch.randn(Hv, device=dev, dtype=torch.float32)
    dt_bias = torch.randn(Hv, device=dev, dtype=torch.float32)
    pool = torch.randn(num_slots, Hv, 128, 128, device=dev, dtype=bf16)  # V-major
    si = torch.arange(N, dtype=torch.int32, device=dev)
    cu = torch.arange(0, total + 1, D, dtype=torch.int32, device=dev)  # [0, D, 2D, ...]
    return q, k, v, a, b, A_log, dt_bias, pool, si, cu


def _rel(x, y):
    return ((x.float() - y.float()).abs().max() / y.float().abs().max().clamp_min(1e-6)).item()


@CUDA
@pytest.mark.parametrize("Hk,Hv", [(8, 8), (2, 8)])
def test_replay_full_tail_matches_verify_commit(Hk, Hv):
    N, D = 4, 8
    q, k, v, a, b, A_log, dt_bias, pool, si, cu = _mk(N, D, Hk, Hv, num_slots=N)
    o = torch.empty(1, N * D, Hv, 128, device="cuda", dtype=torch.bfloat16)
    ibuf = torch.zeros(N, D, Hv, 128, 128, device="cuda", dtype=torch.bfloat16)
    ii = torch.arange(N, dtype=torch.int32, device="cuda")

    # Reference: gated verify WITH commit over the full block.
    pool_ref = pool.clone()
    fused_recurrent_gdr_verify_gated_fwd(
        q, k, v, a, b, A_log, dt_bias, pool_ref, si, cu, ibuf, ii, o,
        disable_state_update=False,
    )

    # Replay: full tail (L=D) from the pre-verify pool, K0 addressing (start=0, stride=D).
    pool_rep = pool.clone()
    seq_idx = torch.arange(N, dtype=torch.int32, device="cuda")
    seq_len = torch.full((N,), D, dtype=torch.int32, device="cuda")
    fused_recurrent_gdr_replay_fwd(
        k, v, a, b, A_log, dt_bias, pool_rep,
        initial_state_indices=si, cache_indices=si,
        input_sequence_indices=seq_idx, input_sequence_lengths=seq_len,
        input_token_start=0, input_token_stride=D,
    )

    rel = _rel(pool_rep, pool_ref)
    print(f"[full-tail Hk={Hk} Hv={Hv}] rel={rel:.2e} equal={torch.equal(pool_rep, pool_ref)}")
    assert rel <= 1e-4, f"replay full-tail state != verify-commit state (rel={rel})"


@CUDA
@pytest.mark.parametrize("L", [1, 3, 5, 8])
def test_replay_partial_tail_matches_verify_intermediate(L):
    N, D, H = 4, 8, 8
    q, k, v, a, b, A_log, dt_bias, pool, si, cu = _mk(N, D, H, H, num_slots=N, seed=1)
    o = torch.empty(1, N * D, H, 128, device="cuda", dtype=torch.bfloat16)
    ibuf = torch.zeros(N, D, H, 128, 128, device="cuda", dtype=torch.bfloat16)
    ii = torch.arange(N, dtype=torch.int32, device="cuda")

    # Verify (no commit) WITH full intermediate cache: ibuf[s, t] = state after t+1 tokens.
    pool_v = pool.clone()
    fused_recurrent_gdr_verify_gated_fwd(
        q, k, v, a, b, A_log, dt_bias, pool_v, si, cu, ibuf, ii, o,
        disable_state_update=True,
    )
    assert torch.equal(pool_v, pool), "no-commit verify must not touch the pool"

    # Replay tail=L -> committed state should equal the verify intermediate at step L.
    pool_rep = pool.clone()
    seq_idx = torch.arange(N, dtype=torch.int32, device="cuda")
    seq_len = torch.full((N,), L, dtype=torch.int32, device="cuda")
    fused_recurrent_gdr_replay_fwd(
        k, v, a, b, A_log, dt_bias, pool_rep,
        initial_state_indices=si, cache_indices=si,
        input_sequence_indices=seq_idx, input_sequence_lengths=seq_len,
        input_token_start=0, input_token_stride=D,
    )
    # ibuf[:, L-1] is V-major [N, H, V, K] == pool slot layout.
    ref = ibuf[:, L - 1]
    got = pool_rep[si.long()]
    rel = _rel(got, ref)
    print(f"[partial L={L}] rel={rel:.2e} equal={torch.equal(got, ref)}")
    assert rel <= 1e-4, f"replay tail={L} state != verify intermediate[L-1] (rel={rel})"


@CUDA
def test_replay_separate_read_write_slots():
    """K0 driver may read from one slot and commit to another (initial != destination)."""
    N, D, H = 3, 6, 8
    q, k, v, a, b, A_log, dt_bias, pool, si, cu = _mk(N, D, H, H, num_slots=2 * N, seed=2)
    o = torch.empty(1, N * D, H, 128, device="cuda", dtype=torch.bfloat16)
    ibuf = torch.zeros(N, D, H, 128, 128, device="cuda", dtype=torch.bfloat16)
    ii = torch.arange(N, dtype=torch.int32, device="cuda")
    read_idx = torch.arange(N, dtype=torch.int32, device="cuda")
    write_idx = torch.arange(N, 2 * N, dtype=torch.int32, device="cuda")

    # Reference: verify-commit reading+writing slot = read_idx.
    pool_ref = pool.clone()
    fused_recurrent_gdr_verify_gated_fwd(
        q, k, v, a, b, A_log, dt_bias, pool_ref, read_idx, cu, ibuf, ii, o,
        disable_state_update=False,
    )

    # Replay: read from read_idx, commit to write_idx.
    pool_rep = pool.clone()
    seq_idx = torch.arange(N, dtype=torch.int32, device="cuda")
    seq_len = torch.full((N,), D, dtype=torch.int32, device="cuda")
    fused_recurrent_gdr_replay_fwd(
        k, v, a, b, A_log, dt_bias, pool_rep,
        initial_state_indices=read_idx, cache_indices=write_idx,
        input_sequence_indices=seq_idx, input_sequence_lengths=seq_len,
        input_token_start=0, input_token_stride=D,
    )
    rel = _rel(pool_rep[write_idx.long()], pool_ref[read_idx.long()])
    print(f"[sep-slots] rel={rel:.2e}")
    assert rel <= 1e-4
    assert torch.equal(pool_rep[read_idx.long()], pool[read_idx.long()]), "read slots untouched"
