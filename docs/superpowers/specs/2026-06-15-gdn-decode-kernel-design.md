# FlashQLA GDN Decode Kernel — Design Spec

Date: 2026-06-15
Status: Design approved in principle (pending written-spec review)
Scope: Forward/inference-only fused **recurrent** (decode) kernel for Gated Delta Rule (GDN), to sit beside the existing chunked-prefill kernels in FlashQLA.

---

## 1. Overview & goals

FlashQLA today ships only the **chunked-prefill** path (`chunk_gated_delta_rule`). This spec adds the **decode** (token-by-token recurrent) path: a memory-bound kernel that advances the GDN recurrent state by `q_len = 1..D` tokens per call and emits the per-token outputs.

Locked requirements (from brainstorming):

1. **General + occupancy-aware.** Run well for both large-batch server decode (`B·H` saturates SMs) and single-request / small-head-count TP (`B·H ≪ SM count`). When `B·H` is small, split each head's state across CTAs to keep SMs busy — the decode analogue of the prefill intra-card CP trick.
2. **Fused multi-step.** `q_len = D` configurable `1..8` (speculative decoding / MTP). Load the `[128,128]` fp32 state once, run all steps with state resident, store once.
3. **Ragged per-sequence accepted length.** In one batched call each of the `B` sequences may commit a different number of tokens `1..D` (speculative verify). A per-sequence length vector drives a per-CTA loop bound.
4. **flash_qla-native API.** A low-level `fused_recurrent_gdr_fwd` python wrapper over the JIT kernel, plus a high-level `recurrent_gated_delta_rule`, mirroring the `chunk/` layering.
5. **Repo-consistent constraints.** `head_dim K=V=128`; q/k/v/o bf16/fp16; state fp32 `[B,H,K,V]`; g/beta fp32; GQA `Hg ≤ H` (`H % Hg == 0`); SM90 (Hopper) only; TileLang 0.1.8; optional `initial_state`, `output_final_state`, `use_qk_l2norm_in_kernel`, and a `scale` arg as in the chunk op.

**Why a separate kernel.** With `chunk_size = 1`, the prefill flow collapses: `A = (I + StrictLower(diag(β)KKᵀ))⁻¹` is a 1×1 `StrictLower = 0`, so `A = I`, and the `kkt_solve` / `W` / `U` machinery and the gate cumsum all vanish. Decode is the bare token recurrence — no `kkt_solve`, no `A`, no cumsum, none of the 512-thread warp-specialized prefill scheduling.

**Regime.** Decode is **memory-bound**: per head per step the math is a couple of length-128 GEMVs and one `128×128` rank-1 update, while the `[128,128]` fp32 state (64 KB/head) dominates HBM traffic. The design optimizes for state I/O, not FLOPs — the opposite of the compute-bound prefill kernel.

---

## 2. The decode recurrence (correctness anchor)

Per `(sequence b, V-head h)`, fp32 state `S ∈ ℝ[K=128, V=128]`. GQA: `group_size = H // Hg`; V-head `h` uses Q/K head `hg = h // group_size` (integer division — `== repeat_interleave(group_size)`; the `mod` mapping is **wrong**, `o_err ≈ 1.8e4`).

Per decode step `t` (token axis), inputs `q_t, k_t ∈ ℝ[K]` from head `hg`; `v_t ∈ ℝ[V]`, scalars `g_t, β_t` (fp32) from head `h`:

```
1.  decay = exp2(g_t · 1.442695)        # raw per-token log-decay g_t ≤ 0; NO cumsum; g=0 → exp2(0)=1 (SWA no-op)
2.  S    ← decay · S                      # gate the WHOLE [K,V] state first
3.  kS   = k_t @ S                         # GEMV over K, reads state AFTER decay → [V]
4.  v_new = β_t · (v_t − kS)               # β on the residual only → [V]
5.  S    ← S + k_t ⊗ v_new                 # rank-1 outer-product update: S[i,j] += k_t[i]·v_new[j]
6.  o_t  = scale · (q_t @ S)               # GEMV on POST-update state → [V]; scale on q only
```

The output reads `S` **after** this token's own decay **and** rank-1 update (post-update / inclusive-diagonal), derived from the kept diagonal (`triu(diagonal=1)`) in `tests/ref_gdr.py::torch_chunk_o_fwd`. Falsified alternatives (re-derived at production `chunk_size=64`): pre-update read `o_err ≈ 7.8e4`; decay-after-update `≈ 2.2e3`; kS-before-decay `≈ 7.5e4`.

Edge cases:
- **`g = 0` (SWA / no-decay) heads:** `decay = 1`, step 2 is a no-op; degenerates to plain (ungated) delta rule. No special path.
- **`scale`:** default `K**-0.5 = 128**-0.5`; honor explicit arg. Applied **only at the output GEMV** (see §7 — folding into q before l2norm cancels).
- **`initial_state`:** if present, load into `S` before the loop; else `S = 0`.

This matches the GDN recurrence in the FlashQLA blog (`S = αS(I − βkkᵀ) + βvkᵀ`) and FLA's `fused_recurrent_gated_delta_rule`.

### Multi-step & ragged
- **`q_len = D` (1..8):** the token axis. Load `S` once, `for t in T.serial(L)`, store `S` once → state HBM traffic is **D-independent** (1 read + 1 write regardless of D). `D=1` is the loop tripping once.
- **Ragged:** runtime `seqlens: [B] int32`, the accepted length `L_b ∈ 1..q_len`. Each CTA reads `L = seqlens[b]` and uses it **directly** as the serial loop bound. Steps `t ≥ L` never execute, so the final state falls out as `S` after the last accepted token, committed by the single post-loop store. Dense `[B, q_len, H, *]` layout (sequences independent; **no** cu_seqlens packing). When `seqlens=None`, the wrapper fills `[B]` with `q_len` (uniform). Wrapper clamps `L ≥ 1`.

---

## 3. Core kernel architecture (`gs = 1`) — build this first

Single-role, memory-bound kernel. **Not** the chunk kernel's 512-thread / 4-warpgroup warp-specialization (that is for a compute-bound chunk pipeline; decode is a serial recurrence on resident state).

- **Threads = 256, single role** (all threads cooperate on every op; the `group_reduce` / `cp_fwd` template, not `fused_fwd`'s producer/consumer split). At 256 threads the `[128,128]` fp32 state fragment is 64 fp32/thread (vs the 128/thread that `threads=128` forces and that the repo never does).
- **Grid** `T.ceildiv(DV, block_DV) · batch_size · H`, 1-D flattened `(bbhv,)`, decoded exactly as `flash_qla/ops/gated_delta_rule/chunk/hopper/fused_fwd.py:90-93`: `bbh, bv = bbhv // ceildiv(DV,block_DV), bbhv % …`; `bb, bh = bbh//H, bbh%H`; `bhg = bh // (H//Hg)`. One CTA owns `(b, V-head, V-column-tile [128, block_DV])` and runs the full `L`-step recurrence on its sub-state end-to-end (no cross-CTA combine).
- **State resident in registers.** `h_fragment = T.alloc_fragment((128, block_DV), "float32")` (the `fused_fwd.py:140` / `prepare_h.py:126` pattern). Loaded once before the loop (`T.copy` from the h0 slice if `use_initial_state` else `T.clear`), mutated in place across all `L` steps, stored once after.
- **GEMM operands live in SMEM.** `gemm_v1` reads SMEM operands and accumulates into a **fragment** (verified: `fused_fwd.py` lines 184/204/254/339 — never accumulates into shared). So each step downcasts the fp32 master to a bf16 operand copy `h_op_shared = T.alloc_shared((128, block_DV), qkva_dtype)` via `T.copy(h_fragment, h_op_shared)` (the `fused_fwd.py:190` fragment→shared copy, which downcasts fp32→bf16). The fp32 master stays in `h_fragment` for the decay/rank-1 accumulation.

### The three K-contractions — all `gemm_v1`
`gemm_v1` is the **only** grounded K-reduction idiom in the repo. (A `T.Parallel` + `reduce_sum(dim=0)`-to-vector reduction over K=128 does **not** exist here — `reduce_sum(dim=0)` appears once, `fused_bwd.py:476`, reducing a 1-D fragment to a scalar — and is rejected.)

- **Step 3 `kS = K @ S`:** `T.gemm_v1(k_op_shared, h_op_shared_decayed, kS_fragment, clear_accum=True)` — mirrors `U = K@S` at `fused_fwd.py:254`.
- **Step 5 rank-1 `S += k ⊗ v_new`:** the **`transpose_A` gemm-into-fragment** `T.gemm_v1(k_op_shared, vn_op_shared, h_fragment, transpose_A=True, clear_accum=False)` — the grounded rank-1 idiom at `fused_fwd.py:204` (Kᵀ@V′ accumulating into the register fragment). *(Correction: `fused_fwd.py:197` is a **scalar-broadcast** decay FMA, **not** a two-vector outer product, so a `T.Parallel(DK,block_DV): h[i,j]+=k[i]·v_new[j]` FMA is **not** grounded — it is a prototype-gated item (§11.E), reserved for the head-batched SMEM-state case where `gemm_v1` cannot accumulate into shared.)*
- **Step 6 `o = Q @ S`:** `T.gemm_v1(q_op_shared, h_op_shared_postupdate, o_fragment, clear_accum=True)`, then `o_fragment *= scale`.
- **Decay (step 2):** `for j_k, j_v in T.Parallel(DK, block_DV): h_fragment[j_k,j_v] *= decay` (`fused_fwd.py:197`).

### Step ordering (critical — post-update read)
Per step: (a) l2norm already done host-side; (b) decay `h_fragment` in place; (c) copy/downcast `h_fragment → h_op_shared`; (d) gemm `kS` on decayed state; (e) `v_new = β·(v − kS)` in fp32; (f) `transpose_A` rank-1 gemm into `h_fragment`; (g) copy/downcast `h_fragment → h_op_shared` again; (h) gemm `o` on post-update state, `*= scale`, cast, store `o[b,t,h, bv-slice]`. After the loop: `if store_final_state: T.copy(h_fragment, final_state[b,h,:, bv-slice])` once.

Note: because each step's `v_new` depends on the running state, the `D` tokens **cannot** be batched into one gemm — all three contractions are **per-step, M=1** (a single token). So §11.A's `M=1` feasibility gate applies to all three, every step.

### M-dim (q_len) and the gemm
Each step is a single token, so the gemm M (token) dim is **1**. **Open feasibility item (§11.A):** confirm `gemm_v1` accepts `M=1` (every repo `gemm_v1` is `M=64`). Mitigation: zero-pad the M (token) dim of `q/k/vn` staging to 16 (zero rows contribute zero to the rank-1 update and produce garbage `o` rows we don't store). Prototype `M=1` first; this gates the whole engine.

---

## 4. Memory & thread layout (core)

- **State HBM layout:** fp32 contiguous `[B,H,128,128]` (the `h0_shape`/`ht_shape` of `fused_fwd.py:70-71`). Innermost V is unit-stride, so the column slice `[b,h, 0:128, bv·block_DV:(bv+1)·block_DV]` is a coalesced / auto-vectorized `T.copy` target. Decode `B = real_batch_size`, one state row per sequence.
- **CTA tile:** `[128, block_DV]`, `block_DV ∈ {128,64,32}`.
- **Register budget:** `block_DV=128 → 64` fp32 state regs/thread at 256 threads; plus `kS_fragment`, `o_fragment`, `v_new` (few/thread each). Target `nreg ≈ 128–160` via `T.set_max_nreg` if needed. `block_DV=64/32` drops pressure 2×/4×. Do **not** materialize a full `[128,block_DV]` product fragment.
- **SMEM:** `h_op_shared [128, block_DV]` bf16 (≤ 32 KB); per-step staging `q/k` (`block_S_pad × 128` bf16, ~4 KB), `v` (`block_S_pad × block_DV`). Total < 64 KB even at `block_DV=128`. Bank conflicts: pad small staging tiles' trailing dim by +1 (the `cumsum.py:47` / `kkt_solve` `17=16+1` idiom) and `T.use_swizzle(10)` (`fused_fwd.py:160`).
- **Reduction-free across CTAs:** because we split V (not K), every CTA holds all 128 K-rows for its V-columns, so the only K-sum (`kS`, `o`) is fully resident. No atomics, no grid-sync, no stitch kernel.

---

## 5. Occupancy strategy (V-split only)

The decode analogue of `cp_context.py`'s auto-CP, computed **host-side** in the wrapper using the `fused_fwd.py:602-608` ladder verbatim:

```
TARGET_NUM_CTAS = int(MULTI_PROCESSOR_COUNT * 0.7)        # fused_fwd.py:12 ; H100 132 SM → 92
grid_base = real_batch_size * H
if   grid_base   >= TARGET_NUM_CTAS: block_DV = 128       # 1 CTA / head
elif grid_base*2 >= TARGET_NUM_CTAS: block_DV = 64        # 2 CTAs / head
else:                                block_DV = 32        # 4 CTAs / head
```

- **Why V-split, not K-split:** the V-column split is reduction-free (decay, `v_new`, rank-1, output are all per-V-column; the only K-contraction is fully resident). **K-split is rejected for `L>1`:** step-3 `kS` contracts all K and feeds `v_new` into step `t+1` nonlinearly, so a K-split needs the per-step `kS` partials summed across CTAs *inside every step*; with no atomics/grid-sync, an HBM-scratch + separate reduction kernel runs only after the first kernel completes and cannot feed step `t` of a running fused loop. K-split forfeits multi-step fusion (back to `L` launches) and is correct only at `L=1`. We do **not** use it; V-split alone gives up to 4×.
- **Honest framing:** server `B·H ≥ TARGET` → `block_DV=128`, full SM fill, **bandwidth-bound**. Single-request `B=1, H=16` → `block_DV=32` → 64 CTAs (~48% of SMs); this regime is **latency-bound, not bandwidth-bound** (16 heads × 128 KB = 2 MB moves in < 1 µs, below the launch + serial-`D`-step-dependency-chain floor). V-split here is a latency-hiding / occupancy knob, not a BW lever — do not claim "near bandwidth-bound in both regimes." For extreme `B=1, Hg≤2` TP, accept partial SM fill; the recommended remedy is the **caller** batching speculative requests / using CUDA graphs to amortize launch (owner decision §13).

---

## 6. Numerics

- **fp32 everywhere for state & accumulation** (`accum_dtype="float32"`): `h_fragment`, `kS_fragment`, `o_fragment`, `v_new`, any l2norm sum. q/k/v/o are bf16/fp16, cast to fp32 on load into the math and back to `o_dtype` only on the final `T.copy(o_fragment → o-slice)`. The gemm **operand** copy of the state is bf16 (`h_op_shared`), but the fp32 master drives decay/rank-1. **`kS` must stay fp32 before the subtract** in `v_new = β·(v − kS)` (catastrophic-cancellation safety, matching the chunk reference).
- **Decay via `exp2`:** `decay = T.exp2(g_t · 1.442695)` (`1.442695 = log₂ e`) under `@tilelang.jit(pass_configs={TL_ENABLE_FAST_MATH: True})` — the repo-wide idiom (`fused_fwd.py:236`, `prepare_h.py:182-183`). **g is consumed raw — never cumsum'd** (the chunk-path cumsum is identity at `chunk_size=1`; applying it gives `err ≈ 1.1`).
- **Scale:** baked as a compile-time literal, applied to q **only at the output GEMV**. When `use_qk_l2norm_in_kernel` is on, scale must **not** be folded into q at load — `l2norm(q·scale) == l2norm(q)` cancels it (footgun).
- **l2norm (host-side — DECIDED):** `recurrent_gated_delta_rule` calls `flash_qla.utils.l2norm(q)/l2norm(k)` exactly as `chunk_gated_delta_rule:221-223` (`rsqrt((x·x).sum(-1)+1e-6)`, `eps=1e-6`, cast back to input dtype). Bit-matches the chunk op; avoids the ungrounded in-kernel `T.rsqrt` + row-reduce. No in-kernel l2norm path in v1.
- **Tolerance:** tests compare at the existing harness bars (0.02 relative for `o` and `final_state`), not fp64-idealized `1e-10`.

---

## 7. Head-batched GQA variant (server regime) — build after the core

A compile-time **specialization of the same jit factory** (keys `head_batch: bool`, `group_size: int`, alongside `block_DV`), not a separate kernel — matching how the chunk factory branches its prim_func on `is_varlen` / `is_cp` / `block_DV`. `head_batch=False` traces the core V-split body unchanged.

**Idea.** In GQA, `group_size = H // Hg` V-heads share one Q/K head `hg`. One CTA owns `(b, head-group hg, V-tile bv)` and processes all `group_size` heads `h = hg·group_size + i`. Per step: load shared `q_t, k_t` **once**; per head load `g, β, v` into the head's column band `[i·blockV:(i+1)·blockV]`; decay-scale each band by its own per-column g; `kS = k_t @ S` as **one wide-N gemm** over `N = group_size·blockV`; `v_new` per band; rank-1 update in place; `o = q_t @ S` as one wide-N gemm (post-update); store each head's state to its disjoint slot after the `L`-loop.

**State residency.** `gs≥2` cannot be a register fragment (e.g. `gs2/bv128` = 256 fp32/thread). Batched state lives in **SMEM** as one concatenated fp32 tile `h_state_shared[128, gs·blockV]`; head `i` owns columns `[i·blockV:(i+1)·blockV]`. A bf16 operand tile `h_op_shared[128, gs·blockV]` is kept for the wgmma-capable K-contractions.

**SMEM budget** vs ~227 KB usable (Hopper opt-in). Hard gate `state + op = 1.5 · 128 · gs·blockV · 4 B`, and `gs·blockV ≤ 512`:

| combo | state+op+staging | occ | verdict |
|---|---|---|---|
| gs2/bv128 | ~206 KB | 1 | feasible (force single-buffer q/k) |
| gs2/bv64 | ~104 KB | 2 | feasible |
| gs2/bv32 | ~54 KB | — | feasible |
| gs3/bv64 | ~154 KB | 1 | feasible |
| gs3/bv32 | ~80 KB | — | feasible |
| gs3/bv128 | ~304 KB | — | **hard-reject** |
| gs4/bv128 | 256 KB state alone | — | **hard-reject** |
| gs4/bv64 | ~206 KB | 1 | feasible (single-buffer only) |
| gs4/bv32 | ~104 KB | 2 | feasible — **cleanest** |

**Feasible set = `{gs2:[128,64,32], gs3:[64,32], gs4:[64,32]}`.** `gs ∈ {1,2,3,4}` are all real configs from the benchmark table; `gs=3` (non-power-of-2 `N`) is first-class.

**Two repo-verified constraints (resolved blockers):**
1. **Rank-1 update is always the `T.Parallel(DK, gs·blockV)` FMA** — `gemm_v1` only accumulates into a *fragment*, and the head-batched state lives only in SMEM, so the `transpose_A` rank-1 gemm-into-shared is **unexpressible**. The wide-N win is preserved on the two *read* contractions (`kS`, `o`), which are the throughput-critical ones.
2. **Fused downcast into the decay pass:** the decay `T.Parallel` writes both outputs in one sweep — `h_state_shared` (fp32, updated) and `h_op_shared = cast_bf16(updated)` (the gemm operand) — and the rank-1 FMA likewise refreshes both. This avoids an unverified shared→shared downcast copy (no repo precedent).

**Auto-selection** (host, composes on the core ladder, gates on **`B·Hg`**):
1. `group_size == 1` → always core V-split.
2. Pick the **largest** feasible `blockV` (maximize `N = gs·blockV`) s.t. `(gs,blockV)` is feasible **and** `grid_hb = B·Hg·ceildiv(DV,blockV) ≥ TARGET`.
3. **Occupancy fence:** with `occ_hb = floor(227KB / smem_per_cta)`, forbid head-batch when `occ_hb==1` and the core keeps a full extra wave — unless `D==1` (where `M=1` under-utilization makes the trade worth it).
4. **No-win fence:** if `N = gs·blockV ≤ 128` (no wider than core `bv=128`), or core already runs `bv=128` with `B·H ≥ TARGET`, skip.
5. **`gs4` default `bv32`:** `tiles(32)=4=gs` → `grid_hb = B·Hg·4 = B·H` (core grid exactly) → wide-N (`N=128`) at zero occupancy cost, occ=2. The cleanest honest win.
6. else core V-split.

**Honest perf.** State bytes are **not** reduced (`gs·64KB` total, same as `gs` separate CTAs). q/k operand reuse saves `(gs−1)·512 B/step` — marginal vs the `gs·128·blockV·2 B` bf16 state operand the gemm reads. The wide-N **wgmma** fill is a genuine win **only for `D>1`** (`M=D` fills wgmma rows); at `D=1`, `M=1` is below the wgmma min M-tile (64) and degenerates to FFMA regardless of `N`, so the `D=1` benefit is instruction-issue (1 wgmma vs `gs`) + operand reuse + lower register/SMEM pressure. Marginal-to-negative when: `N` already ≥128; `B·Hg < TARGET`; `occ_hb==1` with a core extra wave; or `D≥4` (M already fills the tensor core). This is a server-regime, above-the-saturation-knee trade.

**Composition.** Only q/k are shared (read-only); everything mutated is head-private (`h = hg·gs + i`). Ragged `L` is per-CTA (all group heads share sequence `b`, hence the same `L` and the same q/k masking); each head reads its own `g/β/v` and writes its own `o`/`final_state` slot. Result is **bit-identical** to `gs` independent core CTAs (the wide-N gemm computes each column as an independent fixed-order K-reduction — concatenation-invariant), **conditional on** the per-band scalar-gather index (§11).

---

## 8. API & integration

New package `flash_qla/ops/gated_delta_rule/fused_recurrent/` (sibling to `chunk/`), SM90-gated at every import boundary exactly like `chunk/__init__.py:10-13`.

```
fused_recurrent/
  __init__.py                       # fused_recurrent_gdr_fwd wrapper + recurrent_gated_delta_rule; SM90 guard; imports l2norm
  hopper/
    __init__.py                     # from .fused_recurrent_fwd import fused_recurrent_gdr_fwd
    fused_recurrent_fwd.py          # @tilelang.jit factory + low-level python wrapper (mirrors fused_fwd.py two-part structure)
```

**JIT factory** (`hopper/fused_recurrent_fwd.py`), all args compile-time specialization keys; `batch_size` and `num_tokens(=q_len)` are `T.dynamic` so one kernel serves `D=1..8` with no recompile:

```python
@tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True})
def tilelang_fused_recurrent_gdr_fwd(
    H, Hg, DK, DV, scale,
    accum_dtype, qkva_dtype, g_dtype, b_dtype, h0_dtype, ht_dtype, o_dtype, seqlen_dtype,
    use_initial_state, store_final_state, has_seqlens,
    head_batch=False, group_size=1, block_DV=128, threads=256,
):  # -> inner @T.prim_func  (kernel name must end *_kernel_kernel per CLAUDE.md)
```

Tensor signature: `q,k: (batch, q_len, Hg, 128)`; `v,o: (batch, q_len, H, 128)`; `g,beta: (batch, q_len, H)` fp32; `h0, final_state: (batch, H, 128, 128)` fp32; `seqlens: (batch,) int32`. Grid `ceildiv(DV,block_DV)·batch·H` (core) or `·batch·Hg` (head-batch).

**Low-level wrapper:**
```python
def fused_recurrent_gdr_fwd(q, k, v, g, beta, scale=None, initial_state=None,
                            output_final_state=True, seqlens=None, use_qk_l2norm_in_kernel=False):
    # infer (B,q_len,Hg,K)=k.shape, (.,.,H,V)=v.shape; assert K==V==128, H%Hg==0; scale = scale or K**-0.5
    # host auto-selection -> (head_batch, group_size, block_DV) per §5/§7
    # use_initial_state = initial_state is not None (alloc empty h0 if None, like fused_fwd:584-587)
    # o = empty_like(v); final_state = empty((B,H,128,128), fp32)
    # materialize seqlens or full-q_len int32[B]; clamp L>=1
    # compile via factory; launch; return (o, final_state or None)
```

**High-level:**
```python
def recurrent_gated_delta_rule(q, k, v, g, beta, scale=None, initial_state=None,
                               output_final_state=True, use_qk_l2norm_in_kernel=False,
                               seqlens=None, head_first=False):
    # assert q.dtype==k.dtype==v.dtype and != fp32; not head_first; v.shape[2] % k.shape[2] == 0; K==V==128
    # scale = k.shape[-1]**-0.5 if None
    # if use_qk_l2norm_in_kernel: q=l2norm(q); k=l2norm(k)   # HOST-side
    # o, final_state = fused_recurrent_gdr_fwd(...); return o.to(q.dtype), final_state
    # NO autograd.Function (inference/forward-only)
```

**Exports:** add `recurrent_gated_delta_rule` to `flash_qla/ops/gated_delta_rule/__init__.py` and re-export at package top level alongside the chunk entry points. Reused utils only: `flash_qla.utils.l2norm`, the `TARGET_NUM_CTAS` ladder, and the standard TileLang primitives already used in `fused_fwd`/`prepare_h`/`cp_fwd`/`group_reduce`. No cu_seqlens packing, no `chunk_offsets`, no `kkt_solve`, no cumsum, no autograd.

---

## 9. Test plan

Mirror `tests/test_gdr.py` (0.02 relative tolerance, 1000-iter stability loop, SWA `g=0` mask, h0 on/off, GQA `Hk<Hv`).

1. **New torch decode reference** `decode_recur` (add to `tests/ref_gdr.py`): a plain loop of the 6 steps per `(b,h)` over `t=0..L_b-1`, GQA via `hg = bh // (H//Hg)`, optional host `l2norm` (eps 1e-6) before scale, optional `initial_state`, per-sequence accepted-length masking, returning `o[B,q_len,H,V]` (only `[:, :L_b]` valid) and `final_state[B,H,128,128]` fp32.
2. **Pin the reference vs the chunk path:** assert `decode_recur` agrees with `chunk_gated_delta_rule_fwd` at **`chunk_size=64`** (not `chunk_size=1` — `pad_and_reshape(A,dim=1)` at `ref_gdr.py:135` hardcodes 64, so the `chunk_size=1` path is mathematically broken for `T>1`). For a length-`L` single sequence, `decode_recur`'s `o[:, :L]` and `final_state` must match the chunk reference on that `L`-prefix to fp64 roundoff.
3. **Kernel vs `decode_recur`** (bf16 io / 0.02): sweep `q_len ∈ {1,3,8}`; ragged `seqlens=[1,5,8]` mixed in one batch (compare only `o[:, :L_b]`; `final_state[b]` vs the length-`L_b` reference); GQA `Hk=2,Hv=4` (verify head `h` reads k-head `h//2`), MQA `Hk=1`, no-GQA `Hk=Hv`; `initial_state` on/off; explicit non-default scale; `g=0` SWA heads. Run with the SWA mask + 1000-iter loop.
4. **Negative controls:** (a) cumsum'ing g must break (`~1.1`); (b) pre-update read must fail (`~7.8e4`), decay-after-update must fail (`~2.2e3`); (c) `mod` GQA mapping must fail (`~1.8e4`); (d) garbage in `t≥L` positions must change committed `final_state` / `o[:, :L]` by **exactly 0.0**.
5. **Head-batch tests:** **bit-identity** (not 2% tol) of head-batch vs `gs` independent core CTAs with **per-head-distinct g/β** (catches a decay/β band-gather bug); explicit `gs=3` (non-pow2 `N=96/192`) compile + numeric; static-reject asserts that `gs4/bv128` and `gs3/bv128` never reach the JIT.
6. **Feasibility smoke tests** (before full sign-off): compile+run a one-step `M=1` kernel (and `M`-padded-16 fallback) vs `decode_recur` at `L=1`; confirm `T.serial(L)` lowers with runtime per-CTA `L` in the single-role form; confirm the `[128,block_DV]` fp32 fragment compiles at `threads=256` for `block_DV ∈ {128,64,32}` without spill (check nreg).
7. **Signature test** mirroring `tests/test_function_signature.py`: assert the `recurrent_gated_delta_rule` / `fused_recurrent_gdr_fwd` signatures and the `(o, final_state)` return contract.

---

## 10. Benchmark plan

Add `benchmark/bench_recurrent_gdr.py` mirroring `bench_gated_delta_rule.py`. Baseline: FLA `fused_recurrent_gated_delta_rule`, plus `D` separate `D=1` calls (to quantify the fusion win). Use `flash_qla.utils.profile`.

- **Report** wall time and achieved HBM GB/s (% of ~3.35 TB/s peak) **per regime** — not a single "near-peak" claim. Roofline at `B·H ∈ {4,16,64,512,2048}` so the bandwidth claim is falsifiable: server `B·H ≥ TARGET` near peak (state I/O is 94–99% of bytes; time ~ `B·H·128KB/BW`, D-independent); single-request `B·H ≪ TARGET` latency-bound (report launch + serial-`D`-chain time, not bytes/BW).
- **Sweep:** (1) `B ∈ {1,8,64,256}`, `H ∈ {4,16}`, `Hg ∈ {H, H/2, 1}`; (2) `D ∈ {1,4,8}` to show D-fold state-I/O amortization (plot per-token effective state bytes = 128KB/D); (3) `block_DV` auto-selection across the `B·H` sweep; (4) ragged vs uniform `seqlens` (verify ragged adds no measurable overhead); (5) head-batch on/off across `gs ∈ {2,3,4}` to validate the auto-selection win/skip.

---

## 11. Open feasibility items (prototype-gated; do not block the architecture)

These pick between grounded fallbacks; the architecture holds regardless.

- **A. `gemm_v1` at `M=1`** (every step is a single token; every repo `gemm_v1` is `M=64`). The **root gate** — all three contractions (`kS`, rank-1, `o`) are `M=1`. Mitigation: zero-pad the M dim to 16 (zero rows contribute zero; discard garbage `o` rows). Compile-test `M=1` before any other work.
- **B. `T.serial(L)` with a runtime per-CTA `L` on a single-role kernel.** Grounded in `prepare_h.py:166`, but that kernel is warp-spec; confirm in the single-role form. **Static fallback:** `T.serial(q_len)` with an `if t<L` predicate guarding decay+kS+update+store together, and the wrapper zero-fills `g` (decay=1) and `β` (rank-1=0) for `t≥L` (the `fused_fwd.py:406-444` masked-tail idiom).
- **C. Exact nreg at `threads=256, block_DV=128`.** Compile + `set_max_nreg` tuning; confirm no spill.
- **D. Head-batch per-band per-column scalar gather** (decay/β indexed by `band = j_v // blockV` inside one `T.Parallel(DK, gs·blockV)`). **No repo precedent** — existing per-column FMAs index by *row* (`g_exp_shared[j_s]`), uniform across columns. Trace-test that it lowers to a clean per-column broadcast, not a divergent branch. Blocks the head-batch variant if it lowers badly.
- **E. Head-batch fused-downcast pass** (one `T.Parallel` writing both fp32 state and bf16 operand). If it won't compile, fall back to verifying a shared→shared downcast `T.copy` (also unproven).
- **F. fp32-operand wgmma** (`kkt_solve.py:184` proves all-fp32 SMEM operands are legal, but only at 32×32 into a fragment — wgmma-at-scale unproven). Keep bf16 `h_op_shared` default; only elide it (recover ~64 KB) if a microbench shows fp32-A throughput holds. `gs4/bv128` stays hard-rejected regardless.
- **G. Single-buffer enforcement:** factory must statically forbid q/k double-buffering on the occ=1 combos (`gs2/bv128`, `gs4/bv64`).
- **H. Two-vector outer-product `T.Parallel` FMA** `h[i,j]+=k[i]·v_new[j]` — **not** grounded (`fused_fwd.py:197` is a *scalar*-broadcast FMA). The core rank-1 uses the grounded `transpose_A` gemm-into-fragment (`:204`) instead; this FMA is needed **only** for the head-batched SMEM-state case (where gemm can't accumulate into shared) and must be trace-tested to lower as a clean per-element FMA, not a broadcast-shape error.

---

## 12. Implementation order

1. **Core `gs=1` decode kernel** (threads=256, register-resident state, V-split ladder, `T.serial(L)`, multi-step + ragged) — §3–§6. Validate via §9.1–§9.4, §9.6.
2. **Wrappers + API** (§8), exports, signature test (§9.7).
3. **Benchmark** vs FLA `fused_recurrent` (§10) — confirm the core is bandwidth-bound in the server regime.
4. **Head-batched variant** (§7) as a factory specialization, re-derived against the *real* core register/grid numbers. Validate via §9.5, prototype-gates §11.D–§11.G.

Do **not** start the head-batched variant before the core is built and validated — its budget/grid claims must rest on the core's measured numbers, not the chunk kernel's 512-thread / `CONSUMER_S_NREG=160` figures.

---

## 13. Risks & owner decisions

**Resolved in this spec:** vector `reduce_sum(dim=0)` is ungrounded → `gemm_v1` for all K-contractions; `threads=128`+`[128,128]` fp32 spills → `threads=256`; in-kernel l2norm ungrounded → host l2norm; K-split breaks fusion for `L>1` → V-split only; broken `chunk_size=1` provenance → validate vs `chunk_size=64`; scale-before-l2norm cancels → scale at output GEMV only; single-request is latency-bound → framed honestly; head-batch rank-1-into-shared unexpressible → `T.Parallel` FMA; head-batch state too big for registers → SMEM, `gs4/bv128`+`gs3/bv128` hard-rejected.

**Owner decisions (recorded):**
- l2norm location → **host-side** (decided).
- GQA-group head-batching → **in scope now** (decided; §7).
- Extreme `B=1, Hg≤2` TP → **accept V-split ~48% SM fill; caller batches / CUDA graphs** (decided). Not pursuing an `L=1`-only K-split path.

**Ragged contract (resolved by the SGLang grounding):** the **verify** path receives ragged lengths as **`cu_seqlens`/`query_start_loc [N+1]` with `B=1` flattened varlen** (a *different* CTA→request prologue than this spec's dense `[B] seqlens` — derive `bb`/per-request token ranges device-side à la `kkt_solve.py:245-247`, never `.item()`). The dense `[B] seqlens` form is for the standalone decode follow-on. See the verify spec (`2026-06-15-gdn-verify-sglang-design.md`).

**Verify needs `D` up to 12** (chunk-12 draft), which **exceeds this spec's `1..8` design point** — re-derive `nreg` (§11.C) and the `M=1` gemm behavior (§11.A) at `D=12` with a compiled measurement; do not assume the `1..8` budget carries over (the state fragment is `D`-independent, but the per-token-state writes and the serial dependency chain are not).
