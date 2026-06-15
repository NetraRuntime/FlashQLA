# tests/probes/probe_gemm_m1.py
"""Gate 1: does gemm_v1 accept M=1? If not, M-pad to 16. Root gate for the whole engine."""
import torch, tilelang
import tilelang.language as T

DK = DV = 128


def build(M):  # M = padded token rows (1 to test the gate directly; 16 = fallback)
    @tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True})
    def _k():
        @T.prim_func
        def k(kq: T.Tensor([M, DK], "bfloat16"),
              s:  T.Tensor([DK, DV], "bfloat16"),
              o:  T.Tensor([M, DV], "float32")):
            with T.Kernel(1, threads=256) as _:
                ks = T.alloc_shared((M, DK), "bfloat16")
                ss = T.alloc_shared((DK, DV), "bfloat16")
                of = T.alloc_fragment((M, DV), "float32")
                T.copy(kq, ks); T.copy(s, ss)
                T.gemm_v1(ks, ss, of, clear_accum=True)
                T.copy(of, o)
        return k
    return _k()


def run(M):
    torch.manual_seed(0)
    k = torch.randn(M, DK, device="cuda", dtype=torch.bfloat16)
    s = torch.randn(DK, DV, device="cuda", dtype=torch.bfloat16)
    o = torch.empty(M, DV, device="cuda", dtype=torch.float32)
    try:
        build(M)(k, s, o)
    except Exception as e:
        print(f"M={M}: COMPILE/RUN FAIL {type(e).__name__}: {e}")
        return False
    ref = (k.float() @ s.float())
    err = (o - ref).abs().max().item() / ref.abs().max().item()
    ok = err < 0.02
    print(f"M={M}: rel_err={err:.4f} {'OK' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    m1 = run(1)
    m16 = run(16)
    print("\nDECISION: M=1 usable directly:", m1, "| M-pad-to-16 fallback usable:", m16)
