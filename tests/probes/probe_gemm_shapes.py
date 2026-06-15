# tests/probes/probe_gemm_shapes.py
"""Find a threads + warp-policy config under which the 3 decode gemm shapes all compile.
Shapes (M-padded to 16): kS = [16,128]@[128,128]; o same; rank-1 = [16,128]^T @ [16,128] -> [128,128]."""
import inspect
import torch
import tilelang
import tilelang.language as T

print("gemm_v1 signature:", inspect.signature(T.gemm_v1))
pol = None
for cand in ["GemmWarpPolicy"]:
    if hasattr(T, cand):
        pol = getattr(T, cand)
        print(f"T.{cand}:", [p for p in dir(pol) if not p.startswith("_")])
try:
    from tilelang import GemmWarpPolicy as GWP
    print("tilelang.GemmWarpPolicy:", [p for p in dir(GWP) if not p.startswith("_")])
    pol = pol or GWP
except Exception as e:
    print("no tilelang.GemmWarpPolicy:", e)

DK = DV = 128
MPAD = 16


def build(kind, threads, policy):
    @tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True})
    def _k():
        @T.prim_func
        def k(a: T.Tensor([MPAD, DK], "bfloat16"),
              bb: T.Tensor([DK if kind != "rank1" else MPAD, DV], "bfloat16"),
              c: T.Tensor([MPAD if kind != "rank1" else DK, DV], "float32")):
            with T.Kernel(1, threads=threads) as _:
                as_ = T.alloc_shared(a.shape, "bfloat16")
                bs_ = T.alloc_shared(bb.shape, "bfloat16")
                cf = T.alloc_fragment(c.shape, "float32")
                T.copy(a, as_); T.copy(bb, bs_)
                kw = {} if policy is None else {"policy": policy}
                if kind == "rank1":
                    T.gemm_v1(as_, bs_, cf, transpose_A=True, clear_accum=True, **kw)
                else:
                    T.gemm_v1(as_, bs_, cf, clear_accum=True, **kw)
                T.copy(cf, c)
        return k
    return _k()


policies = [None]
if pol is not None:
    for name in ["Square", "FullRow", "FullCol"]:
        if hasattr(pol, name):
            policies.append((name, getattr(pol, name)))

for kind in ["kS", "rank1"]:
    for threads in [64, 128, 256]:
        for p in policies:
            pname = p[0] if isinstance(p, tuple) else "default"
            pval = p[1] if isinstance(p, tuple) else p
            try:
                build(kind, threads, pval)
                print(f"  {kind} threads={threads} policy={pname}: COMPILE OK")
            except Exception as e:
                msg = str(e).splitlines()[-1][:80]
                print(f"  {kind} threads={threads} policy={pname}: FAIL {msg}")
