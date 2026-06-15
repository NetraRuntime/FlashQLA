# benchmark/bench_head_batch.py
# Head-batched GQA vs per-head, core decode kernel. Isolates the head-batch lever (q/k load
# dedup) in the regime where it could help: high GQA ratio + saturated grid.
#
# MEASURED (H100, B in {64,128,256}): grp=2 (Hk4Hv8) = 0.98-0.99x (neutral); grp=4 (Hk2Hv8)
# = 0.74-0.87x (a real regression). The row-stack trades CTA count (B*H -> B*Hg) and uses
# bigger 512-thread CTAs at grp=4, which outweighs the only saving (q/k LOAD dedup, sub-1%).
# Conclusion: head-batch is neutral-to-worse on this memory-bound kernel -> auto stays OFF
# (default); the flag is kept forceable for experimentation. See the decode spec section 7.
import time
import torch

from flash_qla import recurrent_gated_delta_rule
from flash_qla.utils import l2norm


def mk(B, D, Hk, Hv):
    q = l2norm(torch.randn(B, D, Hk, 128, device="cuda", dtype=torch.bfloat16))
    k = l2norm(torch.randn(B, D, Hk, 128, device="cuda", dtype=torch.bfloat16))
    v = torch.randn(B, D, Hv, 128, device="cuda", dtype=torch.bfloat16)
    g = torch.nn.functional.logsigmoid(torch.randn(B, D, Hv, device="cuda")) / 16
    beta = torch.randn(B, D, Hv, device="cuda").sigmoid()
    return q, k, v, g, beta


def timed(fn, iters=300, warmup=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6  # us


def run():
    sm = torch.cuda.get_device_properties().multi_processor_count
    print(f"device={torch.cuda.get_device_name()} SMs={sm}\n")
    print(f"{'Hk':>3} {'Hv':>3} {'grp':>3} {'B':>5} {'per_head(us)':>13} {'head_batch(us)':>15} {'speedup':>8}")
    for Hk, Hv in [(2, 8), (4, 8)]:
        for B in [64, 128, 256]:
            q, k, v, g, beta = mk(B, 8, Hk, Hv)
            kw = dict(scale=128 ** -0.5, output_final_state=True)
            f_hb = lambda: recurrent_gated_delta_rule(q, k, v, g, beta, head_batch=True, **kw)
            f_ph = lambda: recurrent_gated_delta_rule(q, k, v, g, beta, head_batch=False, **kw)
            f_hb(); f_ph()  # build once (exclude JIT from timing)
            us_ph = timed(f_ph)
            us_hb = timed(f_hb)
            print(f"{Hk:>3} {Hv:>3} {Hv // Hk:>3} {B:>5} {us_ph:>13.1f} {us_hb:>15.1f} {us_ph / us_hb:>7.3f}x")


if __name__ == "__main__":
    run()
