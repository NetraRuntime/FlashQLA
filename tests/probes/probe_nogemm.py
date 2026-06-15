# tests/probes/probe_nogemm.py
"""Probe a gemm-free decode step: state [BV, DK], GEMVs as reductions over the last dim,
rank-1 as a T.Parallel outer product. Verify one step vs manual."""
import inspect
import torch
import tilelang
import tilelang.language as T

print("reduce fns on T:", [x for x in dir(T) if "reduce" in x.lower()])
if hasattr(T, "reduce_sum"):
    print("reduce_sum sig:", inspect.signature(T.reduce_sum))

DK = 128
BV = 32  # a V-tile


def build():
    @tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True})
    def _k():
        @T.prim_func
        def k(
            q: T.Tensor([DK], "bfloat16"),
            kk: T.Tensor([DK], "bfloat16"),
            vv: T.Tensor([BV], "bfloat16"),
            gg: T.Tensor([1], "float32"),
            bb: T.Tensor([1], "float32"),
            s_out: T.Tensor([BV, DK], "float32"),
            o_out: T.Tensor([BV], "float32"),
        ):
            with T.Kernel(1, threads=64) as _:
                q_s = T.alloc_shared([DK], "bfloat16")
                k_s = T.alloc_shared([DK], "bfloat16")
                v_s = T.alloc_shared([BV], "bfloat16")
                S = T.alloc_fragment([BV, DK], "float32")
                prod = T.alloc_fragment([BV, DK], "float32")
                kS = T.alloc_fragment([BV], "float32")
                oo = T.alloc_fragment([BV], "float32")
                vnew = T.alloc_fragment([BV], "float32")
                T.copy(q, q_s); T.copy(kk, k_s); T.copy(vv, v_s)
                T.clear(S)
                decay = gg[0]
                # decay (S starts 0, so no-op here, but keep the op)
                for jv, jk in T.Parallel(BV, DK):
                    S[jv, jk] *= T.exp2(decay * 1.442695)
                # kS[v] = sum_dk k[dk]*S[v,dk]
                for jv, jk in T.Parallel(BV, DK):
                    prod[jv, jk] = k_s[jk] * S[jv, jk]
                T.reduce_sum(prod, kS, dim=1)
                # vnew[v] = beta*(v[v] - kS[v])
                for jv in T.Parallel(BV):
                    vnew[jv] = bb[0] * (v_s[jv] - kS[jv])
                # rank-1: S[v,dk] += k[dk]*vnew[v]
                for jv, jk in T.Parallel(BV, DK):
                    S[jv, jk] += k_s[jk] * vnew[jv]
                # o[v] = sum_dk q[dk]*S[v,dk]
                for jv, jk in T.Parallel(BV, DK):
                    prod[jv, jk] = q_s[jk] * S[jv, jk]
                T.reduce_sum(prod, oo, dim=1)
                T.copy(S, s_out)
                T.copy(oo, o_out)
        return k
    return _k()


torch.manual_seed(0)
q = torch.nn.functional.normalize(torch.randn(DK, dtype=torch.float32), dim=0).bfloat16()
kk = torch.nn.functional.normalize(torch.randn(DK, dtype=torch.float32), dim=0).bfloat16()
vv = torch.randn(BV, dtype=torch.float32).bfloat16()
gg = torch.tensor([-0.05], dtype=torch.float32)
bb = torch.tensor([0.4], dtype=torch.float32)
s_out = torch.empty(BV, DK, dtype=torch.float32, device="cuda")
o_out = torch.empty(BV, dtype=torch.float32, device="cuda")
try:
    build()(q.cuda(), kk.cuda(), vv.cuda(), gg.cuda(), bb.cuda(), s_out, o_out)
    # manual: S[v,dk] = k[dk]*beta*v[v]; o[v] = sum_dk q[dk]*S[v,dk] = (q.k)*beta*v[v]
    S_ref = torch.outer(bb * vv.float(), kk.float()).cuda()  # [BV, DK]
    o_ref = (S_ref * q.float().cuda()[None, :]).sum(-1)
    print("S rel err:", ((s_out - S_ref).abs().max() / S_ref.abs().max()).item())
    print("o rel err:", ((o_out - o_ref).abs().max() / o_ref.abs().max()).item())
    print("o_out[:4]:", o_out[:4].tolist(), "o_ref[:4]:", o_ref[:4].tolist())
except Exception as e:
    import traceback
    traceback.print_exc()
    print("FAIL", type(e).__name__, str(e).splitlines()[-1][:120])
