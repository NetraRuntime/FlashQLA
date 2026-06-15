# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]
"""H2 verify-first probe: the decode/verify recurrence is compute-bound on the per-token K-
reductions (kS, oo over DK=128) + the decay/rank-1 [block_DV,DK] FMAs. This sweeps the decode
kernel factory over (block_DV, threads) at compute-bound large-batch regimes to test the
'reduce tiling / threads tuning' lever and confirm whether the autotuned block_DV=64 @ threads=128
leaves any headroom. Also reports achieved GB/s (state I/O) -- a low % vs ~3.35 TB/s peak confirms
compute-bound (not bandwidth-bound), and the final-state-write on/off delta isolates the I/O tail.
"""
import itertools
import torch

from flash_qla.ops.gated_delta_rule.fused_recurrent.hopper.fused_recurrent_fwd import (
    tilelang_fused_recurrent_gdr_fwd,
)

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


def sweep(B, T, H, store_final_state):
    torch.manual_seed(0)
    Hg = H
    q = torch.randn(B, T, Hg, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, T, Hg, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(B, T, H, 128, device="cuda", dtype=torch.bfloat16)
    g = torch.nn.functional.logsigmoid(torch.randn(B, T, H, device="cuda")) / 16
    beta = torch.randn(B, T, H, device="cuda").sigmoid()
    h0 = torch.empty(B, H, 128, 128, device="cuda", dtype=torch.float32)
    ht = torch.empty(B, H, 128, 128, device="cuda", dtype=torch.float32)
    seqlens = torch.empty(B, dtype=torch.int32, device="cuda")
    o = torch.empty_like(v)
    # bytes: per-token o write (B*T*H*128*2) + final state write (B*H*128*128*4 if store)
    o_bytes = B * T * H * 128 * 2
    st_bytes = B * H * 128 * 128 * 4 if store_final_state else 0
    tot_bytes = o_bytes + st_bytes

    print(f"\n== B={B} T={T} H={H} store_final={store_final_state} ==")
    best = None
    for block_DV, threads in itertools.product([32, 64, 128], [128, 256, 512]):
        if threads < block_DV:
            continue
        try:
            kern = tilelang_fused_recurrent_gdr_fwd(
                H, Hg, 128, 128, 128 ** -0.5,
                accum_dtype="float32", qkva_dtype=q.dtype, g_dtype=g.dtype, b_dtype=beta.dtype,
                h0_dtype=h0.dtype, ht_dtype=ht.dtype, o_dtype=o.dtype, seqlen_dtype=seqlens.dtype,
                use_initial_state=False, store_final_state=store_final_state, has_seqlens=False,
                block_DV=block_DV, threads=threads,
            )
            fn = lambda kern=kern: kern(q, k, v, g, beta, h0, seqlens, o, ht)
            us = _time(fn)
        except Exception as ex:  # noqa: BLE001
            print(f"  block_DV={block_DV:<3d} threads={threads:<3d}  FAIL {str(ex).splitlines()[-1][:48]}")
            continue
        gbs = tot_bytes / (us * 1e-6) / 1e9
        mark = ""
        if best is None or us < best[0]:
            best = (us, block_DV, threads)
            mark = " <-- best"
        print(f"  block_DV={block_DV:<3d} threads={threads:<3d}  {us:8.1f} us  {gbs:6.0f} GB/s "
              f"({100*gbs/1000/PEAK_TBS:4.1f}% peak){mark}")
    print(f"  BEST: block_DV={best[1]} threads={best[2]}  {best[0]:.1f} us  "
          f"(as-built dispatch picks block_DV=64 @ threads=128)")


def main():
    print(f"device: {torch.cuda.get_device_name()}")
    for B, T, H in [(256, 12, 32), (64, 12, 32), (256, 4, 32)]:
        sweep(B, T, H, store_final_state=True)
    # floor isolate: final-state-write on vs off (the per-token o write is always on)
    print("\n== final-state-write cost isolate (block_DV=64,threads=128) ==")
    sweep(256, 12, 32, store_final_state=False)


if __name__ == "__main__":
    main()
