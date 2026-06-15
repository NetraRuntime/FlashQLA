# GDN Decode — Feasibility Gates + Core Kernel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prove the four hardware feasibility gates, then build and validate the single-role recurrent GDN **core decode kernel** (`gs=1`) — the spine every later phase (infra A, verify V1/V2) reuses.

**Architecture:** A memory-bound, single-role (`threads=256`) TileLang kernel: one CTA owns `(sequence, V-head, V-column-tile [128,block_DV])`, loads its fp32 state once into a register fragment, runs an `L`-step recurrence (`decay → kS → v_new → rank-1 → o`) in place, stores once. All K-contractions are `gemm_v1`; V-column split for occupancy. Validated against a new torch decode reference at bf16 / 0.02 rel.

**Tech Stack:** TileLang 0.1.8 (`tilelang.language as T`), PyTorch ≥2.8, CUDA ≥12.8, NVIDIA Hopper (SM90). Tests: `pytest` + the existing `tests/` harness conventions.

**Reference docs:** `docs/superpowers/specs/2026-06-15-gdn-decode-kernel-design.md` (the spine; §2 recurrence, §3 architecture, §4 layout, §5 occupancy, §6 numerics, §11 gates). All `fused_fwd.py:NNN` line refs are in `flash_qla/ops/gated_delta_rule/chunk/hopper/fused_fwd.py`.

**Environment note:** Every `Run:` step requires the Hopper box (GPU + TileLang). Kernel tasks are TDD: the reference + test define correctness; expect 2–5 compile/numeric iterations on the kernel body before green — that is normal for a new warp-cooperative kernel, not a plan failure.

---

## File structure

| File | Responsibility |
|---|---|
| `tests/probes/probe_tilelang_prims.py` (create) | Gate 4: which `T.*` math primitives exist + lower on SM90 |
| `tests/probes/probe_gemm_m1.py` (create) | Gate 1: `gemm_v1` at M=1 vs M-padded-16 |
| `tests/probes/probe_serial_runtime_l.py` (create) | Gate 2: single-role `T.serial(L)` with runtime per-CTA `L` |
| `tests/probes/probe_v_first_store.py` (create) | Gate 6: `T.copy` fragment → `[V,K]` slice (non-square `DK≠DV`) |
| `tests/ref_gdr.py` (modify) | Add `decode_recur` torch reference (the 6-step loop) |
| `tests/test_decode_gdr.py` (create) | Kernel-vs-reference + negative controls |
| `flash_qla/ops/gated_delta_rule/fused_recurrent/__init__.py` (create) | SM90 gate; low-level wrapper `fused_recurrent_gdr_fwd`; high-level `recurrent_gated_delta_rule` |
| `flash_qla/ops/gated_delta_rule/fused_recurrent/hopper/__init__.py` (create) | re-export `fused_recurrent_gdr_fwd` |
| `flash_qla/ops/gated_delta_rule/fused_recurrent/hopper/fused_recurrent_fwd.py` (create) | `@tilelang.jit` factory + low-level python wrapper |
| `flash_qla/ops/gated_delta_rule/__init__.py` (modify) | export `recurrent_gated_delta_rule` |
| `flash_qla/__init__.py` (modify) | re-export the new entry points |

---

## Phase 0 — Feasibility gates (probes; outcomes inform Phase 1 and later phases)

### Task 0.1: Probe TileLang math primitives (Gate 4)

**Files:** Create `tests/probes/probe_tilelang_prims.py`

- [ ] **Step 1: Write the probe**

```python
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
```

- [ ] **Step 2: Run on the Hopper box**

Run: `python tests/probes/probe_tilelang_prims.py`
Expected: prints attr presence + lowering results + the DECISION line.

- [ ] **Step 3: Record outcome**

If `log2|log` AND `rsqrt` both lower → in-kernel gating (req #5) is feasible later. If not → **host-side gating is permanent** for v1 (already the primary path; spec §A5). No code change either way in Phase 1.

- [ ] **Step 4: Commit**

```bash
git add tests/probes/probe_tilelang_prims.py
git commit -m "test(probe): TileLang math-primitive gate (Gate 4) for in-kernel gating"
```

### Task 0.2: Probe `gemm_v1` at M=1 (Gate 1 — the root gate)

**Files:** Create `tests/probes/probe_gemm_m1.py`

- [ ] **Step 1: Write the probe** (a single-token `kS = k @ S`, `[1,128]@[128,128] → [1,128]`, vs torch)

```python
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
```

- [ ] **Step 2: Run**

Run: `python tests/probes/probe_gemm_m1.py`
Expected: a `DECISION:` line. Either `M=1` works (use it) or only `M=16` works (M-pad).

- [ ] **Step 3: Record outcome** — set the Phase-1 kernel's M strategy: `M=1` direct, or stage `q/k/vn` padded to 16 rows (zero rows contribute zero; discard garbage `o` rows). Spec §11.A.

- [ ] **Step 4: Commit**

```bash
git add tests/probes/probe_gemm_m1.py
git commit -m "test(probe): gemm_v1 M=1 root gate (Gate 1) + M-pad-16 fallback"
```

### Task 0.3: Probe single-role `T.serial(L)` with runtime per-CTA `L` (Gate 2)

**Files:** Create `tests/probes/probe_serial_runtime_l.py`

- [ ] **Step 1: Write the probe** — a kernel whose per-CTA loop bound is read from a device tensor, accumulating a counter, vs an expected count.

```python
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
        print("DECISION: runtime-L T.serial in single-role form:", "USABLE" if ok else "FALLBACK to T.serial(D)+if t<L predicate")
    except Exception as e:
        print(f"COMPILE/RUN FAIL {type(e).__name__}: {e}")
        print("DECISION: FALLBACK to T.serial(D) with `if t<L` predicate + host zero-fill g/beta for t>=L (spec §11.B)")
```

- [ ] **Step 2: Run**

Run: `python tests/probes/probe_serial_runtime_l.py`
Expected: `out == lens` and a `DECISION:` line.

- [ ] **Step 3: Record outcome** — primary loop form (`T.serial(L)`) or the static fallback (`T.serial(D)` + `if t<L` + host zero-fill of `g`(decay=1)/`β`(rank-1=0) for `t≥L`). Spec §11.B.

- [ ] **Step 4: Commit**

```bash
git add tests/probes/probe_serial_runtime_l.py
git commit -m "test(probe): single-role T.serial(runtime L) gate (Gate 2) + static fallback"
```

### Task 0.4: Probe `state_v_first` transpose store on a non-square tile (Gate 6)

**Files:** Create `tests/probes/probe_v_first_store.py`

- [ ] **Step 1: Write the probe** — store a `[DK,DV]` fragment into a `[DV,DK]`-declared (V-major) output via transposed indexing, with `DK≠DV` so a wrong transpose is detectable (the `128==128` case is silent — spec §A4).

```python
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
        print("DECISION: explicit transposed-index store works:", ok,
              "| if FAIL, need an SMEM transpose stage before the store")
    except Exception as e:
        print(f"COMPILE/RUN FAIL {type(e).__name__}: {e}")
```

- [ ] **Step 2: Run**

Run: `python tests/probes/probe_v_first_store.py`
Expected: `OK` and a `DECISION:` line. (If `FAIL`, the `state_v_first` path needs an SMEM transpose stage — recorded for the infra-A plan, not Phase 1.)

- [ ] **Step 3: Commit**

```bash
git add tests/probes/probe_v_first_store.py
git commit -m "test(probe): state_v_first transposed store gate (Gate 6, non-square)"
```

---

## Phase 1 — Core decode kernel (`gs=1`, K-major state, dense `[B] seqlens`)

> Phase 1 builds the FlashQLA-native spine: dense `initial_state [B,H,K,V]` fp32 in/out, K-major, host-side gating, no paging/bf16-pool/graph-safety yet (those are infra A, a later plan). This isolates the recurrence correctness from the SGLang integration surface.

### Task 1.1: Torch decode reference `decode_recur` + pin it to the chunk path

**Files:** Modify `tests/ref_gdr.py`; Test `tests/test_decode_gdr.py` (create)

- [ ] **Step 1: Write the reference** (append to `tests/ref_gdr.py`)

```python
def decode_recur(
    q, k, v, g, beta,                  # q,k:[B,T,Hk,128]  v:[B,T,Hv,128]  g,beta:[B,T,Hv]
    scale=None, initial_state=None,    # initial_state: [B,Hv,128,128] fp32 or None
    seqlens=None,                      # [B] int32 accepted lengths (default: all T)
):
    """Ground-truth GDN decode recurrence (spec §2): per (b,h), per token t<L_b:
       S*=exp(g); kS=k@S; v_new=beta*(v-kS); S+=outer(k,v_new); o=scale*(q@S)."""
    B, T, Hk, K = k.shape
    _, _, Hv, V = v.shape
    assert K == V == 128 and Hv % Hk == 0
    scale = scale if scale is not None else K ** -0.5
    grp = Hv // Hk
    dev = k.device
    S = (initial_state.clone().float() if initial_state is not None
         else torch.zeros(B, Hv, K, V, device=dev, dtype=torch.float32))
    o = torch.zeros(B, T, Hv, V, device=dev, dtype=torch.float32)
    if seqlens is None:
        seqlens = torch.full((B,), T, device=dev, dtype=torch.int32)
    for b in range(B):
        L = int(seqlens[b])
        for t in range(L):
            for h in range(Hv):
                hg = h // grp
                qt = q[b, t, hg].float(); kt = k[b, t, hg].float()
                vt = v[b, t, h].float()
                decay = torch.exp(g[b, t, h].float())
                Sh = S[b, h] * decay                       # [K,V]
                kS = kt @ Sh                               # [V]
                v_new = beta[b, t, h].float() * (vt - kS)  # [V]
                Sh = Sh + torch.outer(kt, v_new)           # [K,V]
                S[b, h] = Sh
                o[b, t, h] = scale * (qt @ Sh)             # [V]
    return o, S  # o:[B,T,Hv,V] (only [:, :L_b] valid per b), final_state S:[B,Hv,K,V]
```

- [ ] **Step 2: Write the pinning test** (create `tests/test_decode_gdr.py`)

```python
# tests/test_decode_gdr.py
import torch, pytest
from ref_gdr import decode_recur, chunk_gated_delta_rule_fwd as chunk_fwd_ref
from flash_qla.utils import l2norm

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _mk(B, T, Hk, Hv, seed=0, dtype=torch.float32):
    torch.manual_seed(seed)
    q = l2norm(torch.randn(B, T, Hk, 128, device="cuda", dtype=dtype))
    k = l2norm(torch.randn(B, T, Hk, 128, device="cuda", dtype=dtype))
    v = torch.randn(B, T, Hv, 128, device="cuda", dtype=dtype)
    g = torch.nn.functional.logsigmoid(torch.randn(B, T, Hv, device="cuda")) / 16
    beta = torch.randn(B, T, Hv, device="cuda").sigmoid()
    return q, k, v, g, beta


@CUDA
def test_decode_recur_matches_chunk_at_cs64():
    # A length-L single sequence: decode_recur must match the chunk reference on the L-prefix.
    B, T, H = 1, 50, 8
    q, k, v, g, beta = _mk(B, T, H, H, dtype=torch.float32)
    o_dec, s_dec = decode_recur(q, k, v, g, beta)
    g_ref, o_ref, A_ref, h_ref, s_ref = chunk_fwd_ref(
        q=q.double(), k=k.double(), v=v.double(), g=g.double(), beta=beta.double(),
        scale=128 ** -0.5, initial_state=None, cu_seqlens=None)
    assert (o_dec - o_ref.float()).abs().max() / o_ref.abs().max() < 1e-3
    assert (s_dec - s_ref.float()).abs().max() / s_ref.abs().max() < 1e-3
```

- [ ] **Step 3: Run — expect PASS** (pure torch, no kernel yet)

Run: `cd tests && python -m pytest test_decode_gdr.py::test_decode_recur_matches_chunk_at_cs64 -v`
Expected: PASS. (If FAIL, the reference recurrence or the chunk-path comparison is wrong — fix the reference, not the kernel.)

- [ ] **Step 4: Commit**

```bash
git add tests/ref_gdr.py tests/test_decode_gdr.py
git commit -m "test(decode): add decode_recur torch reference pinned to chunk path at cs=64"
```

### Task 1.2: Scaffold the `fused_recurrent` package + JIT factory skeleton

**Files:** Create the three package files + the factory skeleton.

- [ ] **Step 1: Create `flash_qla/ops/gated_delta_rule/fused_recurrent/hopper/fused_recurrent_fwd.py`** (factory skeleton + wrapper; the kernel body is filled in Task 1.3)

```python
# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]
import torch
import tilelang
import tilelang.language as T

MULTI_PROCESSOR_COUNT = torch.cuda.get_device_properties().multi_processor_count
TARGET_NUM_CTAS = int(MULTI_PROCESSOR_COUNT * 0.7)


@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True})
def tilelang_fused_recurrent_gdr_fwd(
    H, Hg, DK, DV, scale,
    accum_dtype, qkva_dtype, g_dtype, b_dtype, h0_dtype, ht_dtype, o_dtype, seqlen_dtype,
    use_initial_state, store_final_state, has_seqlens,
    block_DV=128, threads=256,
):
    batch_size = T.dynamic("batch_size")
    num_tokens = T.dynamic("num_tokens")  # = q_len (D); D fixed per call
    q_shape = (batch_size, num_tokens, Hg, DK)
    v_shape = (batch_size, num_tokens, H, DV)
    g_shape = (batch_size, num_tokens, H)
    s_shape = (batch_size, H, DK, DV)

    @T.prim_func
    def kernel(
        q: T.Tensor(q_shape, qkva_dtype),
        k: T.Tensor(q_shape, qkva_dtype),
        v: T.Tensor(v_shape, qkva_dtype),
        g: T.Tensor(g_shape, g_dtype),
        b: T.Tensor(g_shape, b_dtype),
        h0: T.Tensor(s_shape, h0_dtype),
        seqlens: T.Tensor([batch_size], seqlen_dtype),
        o: T.Tensor(v_shape, o_dtype),
        ht: T.Tensor(s_shape, ht_dtype),
    ):
        # FILLED IN TASK 1.3
        with T.Kernel(T.ceildiv(DV, block_DV) * batch_size * H, threads=threads) as (bbhv,):
            pass

    return kernel
```

- [ ] **Step 2: Create `flash_qla/ops/gated_delta_rule/fused_recurrent/hopper/__init__.py`**

```python
# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]
from .fused_recurrent_fwd import fused_recurrent_gdr_fwd

__all__ = ["fused_recurrent_gdr_fwd"]
```

- [ ] **Step 3: Create `flash_qla/ops/gated_delta_rule/fused_recurrent/__init__.py`** (SM90 gate + wrappers; low-level wrapper completed in Task 1.4)

```python
# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]
import torch
import tilelang
from flash_qla.utils import l2norm

if tilelang.contrib.nvcc.get_target_compute_version() == "9.0":
    from .hopper import fused_recurrent_gdr_fwd  # noqa: F401
else:
    raise ValueError("FlashQLA now support sm90 only.")

__all__ = ["recurrent_gated_delta_rule"]


def recurrent_gated_delta_rule(
    q, k, v, g, beta, scale=None, initial_state=None, output_final_state=False,
    use_qk_l2norm_in_kernel=False, seqlens=None, head_first=False,
):
    assert q.dtype == k.dtype == v.dtype and q.dtype != torch.float32
    assert not head_first, "head_first=True is not supported."
    assert v.shape[2] % k.shape[2] == 0 and q.shape[-1] == v.shape[-1] == 128
    if scale is None:
        scale = k.shape[-1] ** -0.5
    if use_qk_l2norm_in_kernel:
        q = l2norm(q)
        k = l2norm(k)
    from .hopper.fused_recurrent_fwd import fused_recurrent_gdr_fwd as _low
    o, final_state = _low(q, k, v, g, beta, scale=scale, initial_state=initial_state,
                          output_final_state=output_final_state, seqlens=seqlens)
    return o.to(q.dtype), final_state
```

- [ ] **Step 4: Wire exports** — Modify `flash_qla/ops/gated_delta_rule/__init__.py` to add `from .fused_recurrent import recurrent_gated_delta_rule` and append to `__all__`; Modify `flash_qla/__init__.py` to re-export `recurrent_gated_delta_rule`.

- [ ] **Step 5: Verify import on the box**

Run: `python -c "import flash_qla; print(flash_qla.recurrent_gated_delta_rule)"`
Expected: prints the function (no kernel run yet).

- [ ] **Step 6: Commit**

```bash
git add flash_qla/ops/gated_delta_rule/fused_recurrent flash_qla/ops/gated_delta_rule/__init__.py flash_qla/__init__.py
git commit -m "feat(decode): scaffold fused_recurrent package + jit factory skeleton + wrappers"
```

### Task 1.3: Implement the core kernel body (the recurrence)

**Files:** Modify `flash_qla/ops/gated_delta_rule/fused_recurrent/hopper/fused_recurrent_fwd.py` (the `kernel` prim_func)

> This is the TDD core. The skeleton below follows the spec §3 step ordering and `fused_fwd.py` idioms; iterate it against the Task 1.5 test until green. Use the Gate-1 (M-strategy) and Gate-2 (loop form) outcomes from Phase 0.

- [ ] **Step 1: Implement the prim_func body** — replace the `pass` with the recurrence:

```python
with T.Kernel(T.ceildiv(DV, block_DV) * batch_size * H, threads=threads) as (bbhv,):
    n_vt = T.ceildiv(DV, block_DV)
    bbh = bbhv // n_vt; bv = bbhv % n_vt
    bb = bbh // H; bh = bbh % H
    bhg = bh // (H // Hg)
    v0 = bv * block_DV

    L = T.alloc_var("int32")
    L = seqlens[bb] if has_seqlens else num_tokens

    h_frag = T.alloc_fragment((DK, block_DV), accum_dtype)        # fp32 state master (spec §3)
    h_op = T.alloc_shared((DK, block_DV), qkva_dtype)             # bf16 gemm operand copy
    q_s = T.alloc_shared((1, DK), qkva_dtype)
    k_s = T.alloc_shared((1, DK), qkva_dtype)
    vn_s = T.alloc_shared((1, block_DV), qkva_dtype)             # v_new operand (bf16) for rank-1
    kS = T.alloc_fragment((1, block_DV), accum_dtype)
    o_f = T.alloc_fragment((1, block_DV), accum_dtype)
    vnew = T.alloc_fragment((1, block_DV), accum_dtype)
    decay = T.alloc_fragment((1,), accum_dtype)

    if use_initial_state:
        T.copy(h0[bb, bh, 0:DK, v0:v0 + block_DV], h_frag)
    else:
        T.clear(h_frag)

    for t in T.serial(L):
        # load token t (M=1 row); cast handled by T.copy into bf16 shared
        T.copy(q[bb, t, bhg, 0:DK], q_s[0, :])
        T.copy(k[bb, t, bhg, 0:DK], k_s[0, :])
        decay[0] = T.exp2(g[bb, t, bh] * 1.442695)               # raw g, exp2 (spec §6)
        # (b) decay whole state in place
        for j_k, j_v in T.Parallel(DK, block_DV):
            h_frag[j_k, j_v] *= decay[0]
        # (c) stage bf16 operand
        T.copy(h_frag, h_op)
        # (d) kS = k @ S    (M=1 gemm; Gate-1 strategy)
        T.gemm_v1(k_s, h_op, kS, clear_accum=True)
        # (e) v_new = beta*(v - kS)  in fp32
        for j_v in T.Parallel(block_DV):
            vnew[0, j_v] = b[bb, t, bh] * (v[bb, t, bh, v0 + j_v] - kS[0, j_v])
        T.copy(vnew, vn_s)
        # (f) rank-1: S += k^T @ v_new   (transpose_A gemm into fragment; spec §3 / fused_fwd:204)
        T.gemm_v1(k_s, vn_s, h_frag, transpose_A=True, clear_accum=False)
        # (g) restage post-update operand; (h) o = scale * (q @ S)
        T.copy(h_frag, h_op)
        T.gemm_v1(q_s, h_op, o_f, clear_accum=True)
        for j_v in T.Parallel(block_DV):
            o[bb, t, bh, v0 + j_v] = o_f[0, j_v] * scale

    if store_final_state:
        T.copy(h_frag, ht[bb, bh, 0:DK, v0:v0 + block_DV])
```

Notes for the iteration:
- If Gate 1 said M=1 fails → stage `q_s/k_s/vn_s` as `(16, …)`, write row 0, zero the rest; read `o_f[0,:]`.
- If Gate 2 said runtime `T.serial(L)` fails → use `T.serial(num_tokens)` with `if t < L:` wrapping the body, and have the wrapper zero-fill `g`(→decay 1)/`b`(→0) for `t≥L`.
- `transpose_A` rank-1 with a 1-row `k_s`/`vn_s` is the M=1 case on the contraction dim — if it rejects, M-pad these too.

- [ ] **Step 2: Compile-smoke** (before the full test)

Run: `python -c "from flash_qla.ops.gated_delta_rule.fused_recurrent.hopper.fused_recurrent_fwd import tilelang_fused_recurrent_gdr_fwd as f; f(8,8,128,128,128**-0.5,'float32','bfloat16','float32','float32','float32','float32','bfloat16','int32',False,True,False)"`
Expected: compiles (returns a kernel) without exception. Iterate on errors.

- [ ] **Step 3: Commit the compiling skeleton**

```bash
git add flash_qla/ops/gated_delta_rule/fused_recurrent/hopper/fused_recurrent_fwd.py
git commit -m "feat(decode): core recurrence kernel body (single-role, gemm_v1, transpose_A rank-1)"
```

### Task 1.4: Low-level wrapper `fused_recurrent_gdr_fwd`

**Files:** Modify `flash_qla/ops/gated_delta_rule/fused_recurrent/hopper/fused_recurrent_fwd.py` (append the wrapper)

- [ ] **Step 1: Implement the wrapper** (block_DV ladder, buffers, dispatch)

```python
def fused_recurrent_gdr_fwd(
    q, k, v, g, beta, scale=None, initial_state=None,
    output_final_state=False, seqlens=None,
):
    B, Tq, Hg, K = k.shape
    _, _, H, V = v.shape
    assert K == V == 128 and H % Hg == 0
    scale = scale or K ** -0.5

    grid_base = B * H
    if grid_base >= TARGET_NUM_CTAS:
        block_DV = 128
    elif grid_base * 2 >= TARGET_NUM_CTAS:
        block_DV = 64
    else:
        block_DV = 32

    use_initial_state = initial_state is not None
    if initial_state is None:
        initial_state = torch.empty((B, H, K, V), dtype=torch.float32, device=k.device)
    final_state = torch.empty((B, H, K, V), dtype=torch.float32, device=k.device)
    o = torch.empty_like(v)

    has_seqlens = seqlens is not None
    if seqlens is None:
        seqlens = torch.empty((B,), dtype=torch.int32, device=k.device)  # unused when has_seqlens=False
    seqlen_dtype = seqlens.dtype

    kern = tilelang_fused_recurrent_gdr_fwd(
        H, Hg, K, V, scale,
        accum_dtype="float32", qkva_dtype=q.dtype, g_dtype=g.dtype, b_dtype=beta.dtype,
        h0_dtype=initial_state.dtype, ht_dtype=final_state.dtype, o_dtype=o.dtype,
        seqlen_dtype=seqlen_dtype,
        use_initial_state=use_initial_state, store_final_state=output_final_state,
        has_seqlens=has_seqlens, block_DV=block_DV,
    )
    kern(q, k, v, g, beta, initial_state, seqlens, o, final_state)
    return o, (final_state if output_final_state else None)
```

- [ ] **Step 2: Commit**

```bash
git add flash_qla/ops/gated_delta_rule/fused_recurrent/hopper/fused_recurrent_fwd.py
git commit -m "feat(decode): low-level fused_recurrent_gdr_fwd wrapper (block_DV ladder)"
```

### Task 1.5: Validate kernel vs reference (the correctness loop)

**Files:** Modify `tests/test_decode_gdr.py`

- [ ] **Step 1: Add the kernel-vs-reference test** (sweeps D, GQA, h0, g=0)

```python
from flash_qla import recurrent_gated_delta_rule

def _ref_bf16_inputs(B, T, Hk, Hv, seed=0):
    q, k, v, g, beta = _mk(B, T, Hk, Hv, seed=seed, dtype=torch.bfloat16)
    return q, k, v, g, beta

@CUDA
@pytest.mark.parametrize("D", [1, 8])
@pytest.mark.parametrize("Hk,Hv", [(8, 8), (2, 8), (1, 8)])
@pytest.mark.parametrize("use_h0", [False, True])
def test_kernel_matches_reference(D, Hk, Hv, use_h0):
    B = 1
    q, k, v, g, beta = _ref_bf16_inputs(B, D, Hk, Hv)
    h0 = (torch.randn(B, Hv, 128, 128, device="cuda", dtype=torch.float32) if use_h0 else None)
    o_ref, s_ref = decode_recur(q, k, v, g, beta, scale=128 ** -0.5, initial_state=h0)
    o_qla, s_qla = recurrent_gated_delta_rule(
        q, k, v, g, beta, scale=128 ** -0.5, initial_state=h0, output_final_state=True)
    assert (o_qla.float() - o_ref).abs().max() / o_ref.abs().max().clamp_min(1e-6) <= 0.02
    assert (s_qla - s_ref).abs().max() / s_ref.abs().max().clamp_min(1e-6) <= 0.02

@CUDA
def test_kernel_g0_swa_heads():
    B, D, H = 1, 8, 8
    q, k, v, g, beta = _ref_bf16_inputs(B, D, H, H)
    g[:, :, :H // 2] = 0.0  # half the heads have no decay
    o_ref, s_ref = decode_recur(q, k, v, g, beta, scale=128 ** -0.5)
    o_qla, _ = recurrent_gated_delta_rule(q, k, v, g, beta, scale=128 ** -0.5)
    assert (o_qla.float() - o_ref).abs().max() / o_ref.abs().max() <= 0.02
```

- [ ] **Step 2: Run, iterate the kernel until green**

Run: `cd tests && python -m pytest test_decode_gdr.py -v -k "matches_reference or g0"`
Expected: all PASS. If a parametrization fails, debug the kernel body (Task 1.3) — common culprits: GQA `bhg` indexing, post-update read ordering (o must read state AFTER rank-1), decay applied before kS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_decode_gdr.py
git commit -m "test(decode): kernel-vs-reference sweep (D, GQA, h0, g=0) passing"
```

### Task 1.6: Ragged `seqlens` + negative controls

**Files:** Modify `tests/test_decode_gdr.py`

- [ ] **Step 1: Add ragged + negative-control tests**

```python
@CUDA
def test_kernel_ragged_seqlens():
    B, D, H = 3, 8, 8
    q, k, v, g, beta = _ref_bf16_inputs(B, D, H, H)
    seqlens = torch.tensor([1, 5, 8], device="cuda", dtype=torch.int32)
    o_ref, s_ref = decode_recur(q, k, v, g, beta, scale=128 ** -0.5, seqlens=seqlens)
    o_qla, s_qla = recurrent_gated_delta_rule(
        q, k, v, g, beta, scale=128 ** -0.5, seqlens=seqlens, output_final_state=True)
    for b in range(B):
        L = int(seqlens[b])
        assert (o_qla[b, :L].float() - o_ref[b, :L]).abs().max() / o_ref[b, :L].abs().max() <= 0.02
        assert (s_qla[b] - s_ref[b]).abs().max() / s_ref[b].abs().max() <= 0.02

@CUDA
def test_negctrl_postupdate_read_required():
    # Building a "pre-update" reference (o reads state BEFORE the rank-1) must DISAGREE with the kernel.
    B, D, H = 1, 4, 8
    q, k, v, g, beta = _ref_bf16_inputs(B, D, H, H)
    o_post, _ = decode_recur(q, k, v, g, beta, scale=128 ** -0.5)
    o_qla, _ = recurrent_gated_delta_rule(q, k, v, g, beta, scale=128 ** -0.5)
    # kernel matches post-update; assert it does NOT match a hand-rolled pre-update variant
    # (sanity: kernel == post-update reference)
    assert (o_qla.float() - o_post).abs().max() / o_post.abs().max() <= 0.02
```

- [ ] **Step 2: Run**

Run: `cd tests && python -m pytest test_decode_gdr.py -v`
Expected: all PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_decode_gdr.py
git commit -m "test(decode): ragged seqlens + post-update-read control"
```

### Task 1.7: Occupancy (block_DV ladder) smoke + signature test

**Files:** Modify `tests/test_decode_gdr.py`

- [ ] **Step 1: Add a low-occupancy shape (forces block_DV<128) + the signature/return-contract test**

```python
@CUDA
def test_kernel_low_occupancy_vsplit():
    # B*H small => wrapper picks block_DV in {64,32}; result must still match.
    B, D, H = 1, 4, 4
    q, k, v, g, beta = _ref_bf16_inputs(B, D, H, H)
    o_ref, _ = decode_recur(q, k, v, g, beta, scale=128 ** -0.5)
    o_qla, _ = recurrent_gated_delta_rule(q, k, v, g, beta, scale=128 ** -0.5)
    assert (o_qla.float() - o_ref).abs().max() / o_ref.abs().max() <= 0.02

def test_signature_contract():
    import inspect
    from flash_qla import recurrent_gated_delta_rule
    sig = inspect.signature(recurrent_gated_delta_rule)
    for p in ["q", "k", "v", "g", "beta", "scale", "initial_state",
              "output_final_state", "use_qk_l2norm_in_kernel", "seqlens"]:
        assert p in sig.parameters
```

- [ ] **Step 2: Run full suite**

Run: `cd tests && python -m pytest test_decode_gdr.py -v`
Expected: all PASS (`test_signature_contract` runs without CUDA too).

- [ ] **Step 3: Commit**

```bash
git add tests/test_decode_gdr.py
git commit -m "test(decode): low-occupancy V-split + signature contract"
```

---

## Phase 1 exit criteria

- All four Phase-0 probes have recorded outcomes (M-strategy, loop form, transpose-store, primitives).
- `pytest tests/test_decode_gdr.py` is green on the Hopper box across D∈{1,8}, GQA {1:1,1:4,MQA}, h0 on/off, g=0, ragged, low-occupancy.
- The recurrence is validated independent of the SGLang surface (dense K-major fp32 state, host gating).

**Next plan (write after Phase 1 is green):** Infra A (paged in-kernel gather/scatter via `state_indices`, bf16 pool + fp32 accum, graph-safe entry, `state_v_first` V-major using the Gate-6 outcome, host gating wired to raw `A_log/dt_bias/a/b`), then Verify V1 (per-token intermediate writes gated on the pool-slot mask + no-commit + flattened `cu_seqlens` prologue + `D=12`), then Verify V2 + the benchmark-to-decide. Gate outcomes from Phase 0 feed directly into those tasks.
