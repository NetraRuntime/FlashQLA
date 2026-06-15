# tests/probes/probe_tilelang_prims.py
"""Gate 4: which TileLang math intrinsics exist and lower on SM90.
Decides in-kernel gating feasibility (softplus needs log/log2; l2norm needs rsqrt)."""
import tilelang
import tilelang.language as T

NAMES = ["exp2", "exp", "log", "log2", "log1p", "rsqrt", "sqrt", "sigmoid", "tanh", "pow", "abs"]


def report_attrs():
    have = {n: hasattr(T, n) for n in NAMES}
    print("attr presence:", have)
    return have


def lower_smoke(name):
    """Try to actually lower a 1-op kernel using T.<name>; return True if it compiles."""
    fn = getattr(T, name, None)
    if fn is None:
        return False

    @tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True})
    def _k():
        @T.prim_func
        def k(x: T.Tensor([128], "float32"), y: T.Tensor([128], "float32")):
            with T.Kernel(1, threads=128) as _:
                for i in T.Parallel(128):
                    y[i] = fn(x[i])
        return k

    try:
        _k()  # JIT/compile
        return True
    except Exception as e:
        print(f"  lower {name}: FAIL {type(e).__name__}: {e}")
        return False


if __name__ == "__main__":
    have = report_attrs()
    print("lowering:")
    lowered = {n: (lower_smoke(n) if have[n] else False) for n in NAMES}
    print("lowered:", lowered)
    print("\nDECISION: in-kernel gating feasible iff log2(or log)+rsqrt both lower:",
          (lowered.get("log2") or lowered.get("log")) and lowered.get("rsqrt"))
