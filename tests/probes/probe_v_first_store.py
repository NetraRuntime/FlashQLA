# tests/probes/probe_v_first_store.py
"""Gate 6: does T.copy / indexed store from a [DK,DV] fragment into a V-major [DV,DK] slice
emit a correct strided store? Use DK!=DV so a transpose bug is NOT numerically silent."""
import torch, tilelang
import tilelang.language as T

DK, DV = 128, 64  # deliberately non-square


def build():
    @tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True})
    def _k():
        @T.prim_func
        def k(src: T.Tensor([DK, DV], "float32"),
              dst: T.Tensor([DV, DK], "float32")):   # V-major destination
            with T.Kernel(1, threads=256) as _:
                f = T.alloc_fragment((DK, DV), "float32")
                T.copy(src, f)
                for i, j in T.Parallel(DK, DV):
                    dst[j, i] = f[i, j]                # transposed store
        return k
    return _k()


if __name__ == "__main__":
    src = torch.randn(DK, DV, device="cuda", dtype=torch.float32)
    dst = torch.empty(DV, DK, device="cuda", dtype=torch.float32)
    try:
        build()(src, dst)
        ok = torch.allclose(dst, src.t(), atol=1e-5)
        print("transpose store:", "OK" if ok else "FAIL (max diff %.3e)" % (dst - src.t()).abs().max())
        print("DECISION: explicit transposed-index store works:", bool(ok),
              "| if FAIL, need an SMEM transpose stage before the store")
    except Exception as e:
        print(f"COMPILE/RUN FAIL {type(e).__name__}: {e}")
