# tests/probes/probe_serial_runtime_l.py
"""Gate 2: single-role threads=256 kernel with a runtime per-CTA loop bound L=lens[bb]."""
import torch, tilelang
import tilelang.language as T


def build():
    @tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True})
    def _k():
        B = T.dynamic("B")
        @T.prim_func
        def k(lens: T.Tensor([B], "int32"), out: T.Tensor([B], "float32")):
            with T.Kernel(B, threads=256) as (bb,):
                Lv = T.alloc_var("int32"); Lv = lens[bb]
                acc = T.alloc_fragment((1,), "float32"); acc[0] = 0.0
                for _t in T.serial(Lv):
                    acc[0] += 1.0
                out[bb] = acc[0]
        return k
    return _k()


if __name__ == "__main__":
    lens = torch.tensor([1, 5, 12, 8], device="cuda", dtype=torch.int32)
    out = torch.empty(4, device="cuda", dtype=torch.float32)
    try:
        build()(lens, out)
        ok = torch.allclose(out, lens.float())
        print("out:", out.tolist(), "expected:", lens.tolist(), "->", "OK" if ok else "FAIL")
        print("DECISION: runtime-L T.serial in single-role form:",
              "USABLE" if ok else "FALLBACK to T.serial(D)+if t<L predicate")
    except Exception as e:
        print(f"COMPILE/RUN FAIL {type(e).__name__}: {e}")
        print("DECISION: FALLBACK to T.serial(D) with `if t<L` predicate + host zero-fill g/beta for t>=L (spec 11.B)")
