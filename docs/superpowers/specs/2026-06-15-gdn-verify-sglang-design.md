# FlashQLA GDN Verify + SGLang Integration — Design Spec

Date: 2026-06-15
Status: Design approved in principle (pending written-spec review)
Depends on: `2026-06-15-gdn-decode-kernel-design.md` (the recurrent decode **spine** is a hard prerequisite).
Grounding: verified against SGLang upstream `f18d38d` (via `gh api` blob fetch) and a grep of this repo's TileLang usage.

> **IMPLEMENTATION STATUS (2026-06-15, validated on a Modal H100).** V1 verify is **built and passing** (30 decode+verify tests at 0.02 rel). Key deviation from this spec: the kernel is **gemm-free** — the GEMVs are `T.reduce_sum` over the K dim and the rank-1 is a `T.Parallel` outer product, with the state kept `[block_DV, DK]` (V-major) in an fp32 fragment (Gate 1 showed `gemm_v1` needs M%16==0 *and* num_warps≤N/16, which is unworkable for single-token GEMVs at small `block_DV`; the gemm-free path is simpler and keeps the state fp32). Because the state is already V-major, the SGLang V-major pool store is a **direct** write (no transpose). Delivered: paged `state_indices` gather/scatter (slot<0 skip), bf16 pool, per-token intermediates (gated on the pool mask), no-commit, varlen `cu_seqlens`, host **and** in-kernel fused gating (`A5` — Gate 4 showed `log/log1p/rsqrt/sigmoid/exp` lower on SM90, so it is **not** prototype-gated), and CUDA-graph safety. **Benchmark:** server regime (`N≥64,H=32`) is bandwidth-bound at **60–67% of peak HBM**; single-request TP8 (`N=1,H=8`) is latency-bound at ~6% (as predicted — caller batches / CUDA graphs). **V2 (chunk-based) is NOT built:** the benchmark confirms V1 is bandwidth-bound on the inherent per-token state writes, which V2 cannot avoid while adding chunk+kkt overhead — so V2 cannot win the total-latency metric. Files: `flash_qla/ops/gated_delta_rule/fused_recurrent/`; tests `tests/test_decode_gdr.py`, `tests/test_verify_gdr.py`; bench `benchmark/bench_recurrent_gdr.py`.

---

## 1. Overview & scope

Extend FlashQLA to serve **SGLang speculative decoding** for GDN: a CUDA-graph-safe, paged, bf16-state **verify** kernel that emits per-token outputs **and** per-token intermediate states, without committing the final state. Decode (T=1) is a later/optional follow-on; this spec is verify-first.

**Resolved scope (do not relitigate):**
- **Verify-first.** Build the shared infra (A) on top of the decode spine, then the verify kernel (B). Standalone single-token decode (C) is a follow-on; the verify engine already covers the multi-token recurrence.
- **Build BOTH verify engines and benchmark at T=12** (decision): **V1** = recurrent multi-step (per-token states native); **V2** = chunk-based (fast `o`) + per-token-state extraction. Ship the faster on **total verify-step latency (`o` + per-token states)**. The repo-grounded analysis predicts V1 wins (see §5); V2 is built to confirm with data.
- **Linear draft chain only** — target scheme is **DFlash** (public upstream SGLang linear-chain speculative verify). Tree-structured propagation (DDTree) is **out of scope** (not needed); the linear design doesn't preclude adding it later.
- **`allow_neg_eigval = False`** (decision): plain `β = sigmoid(b)`, exposed as a compile-time flag defaulting False.
- **bf16 state pool** (your deployment; `SGLANG_MAMBA_SSM_DTYPE`), kernel dtype-agnostic.
- **Scope boundary.** This kernel owns **only the GDN SSM recurrent state**. The conv state (`causal_conv1d_update` — tree-conv + `intermediate_conv_window`) runs **upstream in SGLang** (`gdn_backend.forward_extend`), before this kernel — it is **not** this kernel's responsibility. The author owes no conv-state handling.

**The root gate for everything** is decode-spec §11.A: `gemm_v1` at `M=1` (each step is one token). Prove it on the Hopper box before any verify work; fallback is M-pad-to-16.

---

## 2. Shared infra A (the reusable base for V1, V2, and decode)

Infra A is a set of **compile-time keys + kernel-arg contracts + code hooks** the recurrent decode kernel (and the chunk kernel, for V2) gains. It is **not** five edits to an existing factory — the recurrent factory does not exist yet; A is built *with* the spine (decode-spec §12 order).

### A1 — Paged in-kernel gather/scatter (GROUNDED, ship)
State lives in a caller-owned pool `[num_slots, H, V, K]` (V-major — see A4) indexed per request by `state_indices: [N] int32`. The gather/scatter is **in-kernel** — Python never slices `pool[idx]`. Idiom verified at `cp_fwd.py:141` (`cp_h0[seq_start_idx, bh, …]` with `seq_start_idx` an `int32` `alloc_var` loaded from a device tensor) and the double-indirection at `kkt_solve.py:245-247`.

Per CTA owning `(request b, V-head bh, V-tile bv)`:
```
slot = T.alloc_var('int32'); slot = state_indices[bb]
T.clear(h_fragment)                                  # clear first
with T.If(slot >= 0):                                 # NO T.Else exists in the repo
    with T.Then():
        T.copy(pool[slot, bh, <state-slice>], h_fragment)   # bf16 → fp32 gather
… recurrence …
if not disable_state_update:
    with T.If(slot >= 0):
        with T.Then():
            T.copy(h_fragment, pool[slot, bh, <state-slice>])   # fp32 → bf16 scatter
```
**Clear-then-conditional-overwrite** (there is no `T.If/T.Else` in the repo). **Only `state_indices` carries the `-1` sentinel** (`PAD_SLOT_ID = -1`; `slot < 0 ⇒ skip`, slot 0 is valid — **not** vLLM's `slot ≤ 0 / NULL_BLOCK_ID=0`); guard with `slot >= 0` (resolved §10.4). **`intermediate_state_indices` is dense (`arange`, never `-1`)** — so the per-token ibuf write (§3) must be gated by the **pool-slot mask `state_indices[bb] >= 0`**, reusing the same `slot` var, **not** by `intermediate_state_indices`. (A `T.If(cache_slot >= 0)` guard would never fire — `cache_slot` is always valid — and would leave garbage in padded ibuf rows.)

### A2 — Configurable-dtype pool + fp32-register accumulation (GROUNDED, ship)
fp32 master `h_fragment = alloc_fragment((128, block_DV), "float32")`. **Exactly three cast points:** (a) bf16→fp32 on the gather; (b) fp32→bf16 on the per-step gemm-operand staging copy (`fused_fwd.py:190`); (c) fp32→bf16 on intermediate/final stores. The cancellation-critical `v_new = β·(v − kS)` subtract stays **fp32** before any downcast. Pool dtype is a factory key driven by the caller's pool tensor dtype (your deployment: bf16). Under fp32 pools the per-token-write traffic doubles (~832 KB/head vs ~416 KB/head) — a caller decision, not a code risk.

### A3 — Graph-safe entry (GROUNDED, ship)
Three rules for the captured call:
1. **No host sync.** Never call `prepare_chunk_offsets` (`index.py:138` ends in `.item()`) or `prepare_chunk_indices` (`index.py:80` `.tolist()`). Ragged length comes from a device `cu_seqlens` consumed via in-kernel `alloc_var` load (`kkt_solve.py:39,247`), never `.item()`'d.
2. **No allocation.** Every buffer (`o`, pool, intermediate buffer, all index tensors) is **caller-preallocated**; `out_idx` stays commented (`fused_fwd.py:16`) so outputs are positional. The graph-safe wrapper does **zero** `torch.empty` (vs `fused_gdr_fwd:561-600`).
3. **Static shapes.** `block_DV` from the `MULTI_PROCESSOR_COUNT*0.7` ladder (`fused_fwd.py:602-608`) using only static `N·H`; `grid = ceildiv(DV,block_DV)·N·H`; `T=12` fixed per graph; `cache_steps` read from `intermediate_states_buffer.shape[1]` as a python int (capture-constant), **not** the runtime `cache_steps` arg (SGLang ignores it).

**"Host-side" means PyTorch, not TileLang — NOT outside capture.** The gating + l2norm (A5) depend on per-step `a, b, q, k`, so they are PyTorch ops that run **inside** SGLang's captured graph; they are capture-safe (pure elementwise, no `.item()`, no new allocation, static shape) and must **not** be lifted out of the per-step graph.

### A4 — State layout: `state_v_first = True` is the SGLang contract (GROUNDED, default flipped)
SGLang's pool **and** intermediate buffer are **V-major `[.,H,V,K]`** — established from pointer **arithmetic** (`o_v*K + o_k`; `make_block_ptr (V,K),(K,1)`; `temporal_state_shape=(HV, head_dim=V, state_size=K)`), **not** docstrings (which are stale and mutually contradictory). FlashQLA-native is K-major `[.,H,K,V]` (`fused_fwd.py:70`). `state_v_first` is a compile-time key tracing to a different prim_func body (like `is_varlen`/`is_cp`); **no runtime transpose** (incompatible with paging). It applies to **both** the pool and the intermediate buffer (the scheduler reads the buffer back; a wrong major-order silently restores a transposed state). The wrapper derives it authoritatively from `pool.stride()/shape`.

**Critical test consequence:** because `K==V==128`, a layout error is **numerically silent** in any equal-dim test. The hard validation gate is a **per-head-distinct-gate bit-identity** test vs the FLA reference (§8), never a shape/equal-value test. Whether `T.copy` from a `[K, block_DV]` fp32 fragment into a `[V,K]`-declared slice emits a strided TMA store without an SMEM transpose stage is **prototype-gated** (§9 Gate 6) — validate on a non-square probe (`DK≠DV`) or byte-compare with FLA, never the `128==128` test.

### A5 — Gating: host-side primary, in-kernel prototype-gated
The repo's TileLang uses **only `T.exp2`** (grep: zero `log`/`log2`/`log1p`/`rsqrt`/`sqrt`/`sigmoid`/`exp`). In-kernel `softplus` needs `log1p`; in-kernel l2norm needs `rsqrt` — **neither is grounded**, and the decode spec already decided host-side l2norm. The Gate-4 probe (`hasattr(T,'log2'/'rsqrt')` + an SM90 lower test) could not run here (no TileLang locally) — it **must run on the Hopper box** (§9 Gate 4).

- **Primary (capture-safe, ships):** compute `g`, `β`, and qk-l2norm in **PyTorch (not TileLang)** — a tiny `[1, N·T, H]` elementwise op + l2norm that runs **inside** SGLang's captured graph (capture-safe), passing **pre-activated `g` (log-decay) and `β` (post-sigmoid)** into the kernel — exactly what SGLang's `_update` kernel and FlashQLA's chunk path already consume. Only the per-step decay `exp2(g·1.442695)` is in-kernel (pure `exp2`, grounded).
- **In-kernel fusion (req #5, fast-follow):** accept raw `(A_log, dt_bias, a, b)` and compute `g`/`β`/l2norm in-kernel — **only if** Gate 4 confirms `log2`/`rsqrt` lower on SM90. `sigmoid` and the decay `exp2` are pure-`exp2` (groundable); `softplus`/l2norm-`rsqrt` are the blockers.

### Exact gating math (grounded — identical across vLLM / SGLang / FLA)
```
softplus(x) = log(1 + exp(x))  for x ≤ 20 (threshold), else x   # softplus_beta=1.0
g  = -exp(A_log) · softplus(a + dt_bias)        # g ≤ 0 (log-decay). A_log = log(A), A ~ U(0,16)
β  = sigmoid(b)                                  # allow_neg_eigval=False (decision); if True, β *= 2
l2norm: q = q / sqrt(Σ q² + 1e-6) ;  k = k / sqrt(Σ k² + 1e-6)   # eps INSIDE the sqrt; fp32; per token,head
then:  q *= scale  (scale = 128**-0.5, q only; k NOT scaled)     # AFTER l2norm — folding before cancels
decay applied per step:  S *= exp(g)             # RAW g, never cumsum'd in the recurrent path
```
Reference-fixture init (for tests): `A ~ U(0,16)`, `A_log = log(A)`; `dt = exp(U(log 1e-3, log 1e-1))` clamped `≥1e-4`; `dt_bias = inv_softplus(dt)`.

---

## 3. Verify V1 — recurrent multi-step + per-token intermediates (the shipping per-token path)

**Prerequisite chain (hard gate, decode-spec §12):** build (1) the core `gs=1` decode kernel, validated; then (2) paging + no-commit + per-token-intermediate writes; then (3) optional in-kernel gating. The two unproven primitives blocking (1) are §11.A (`M=1` gemm) and §11.B (single-role `T.serial(L)` with runtime `L`) — prototype both **first**.

**Per-token intermediates (req #6) — the free V1 win (GROUNDED).** The serial loop already holds the full post-update fp32 `S` in `h_fragment` at the end of every token (step ordering: decay → stage → `kS` → `v_new`(fp32) → rank-1 → **write state** → `o`). The write:
```
if store_intermediate:
    with T.If(slot >= 0):                                            # gate on the POOL slot mask
        with T.Then():
            T.copy(h_fragment, ibuf[cache_slot, t, bh, <state-slice>])   # fp32 → bf16
```
`cache_slot = intermediate_state_indices[bb]` is the **destination** index (decoupled from `state_indices`); it is dense (`arange`, never `-1`), so the write is gated by the **pool-slot mask `slot = state_indices[bb] >= 0`** — the real-request mask — reusing A1's `slot` var. (Guarding on `cache_slot >= 0` would never fire and would write garbage into padded ibuf rows; harmless only because the scheduler reads `ibuf[slot, :k_accepted]` for real requests, but we skip those writes anyway.) `ibuf = intermediate_states_buffer [num_slots+1, cache_steps=12, H, V, K]` (V-major, single-layer slice — see §6). This is `fused_fwd.py:471-481` with `batch_idx → cache_slot` (paged) and `chunk_start_idx+i_s → token t`. Cost: +1 bf16 `[128,block_DV]` store/token; ~`12·32KB = 384 KB/head` of writes dominate (~92% of state traffic) — **inherent to req #6, the workload, not overhead.**

**No-commit (req #7) — free (GROUNDED).** The spine's only pool write is the optional post-loop scatter (A1), gated by `not disable_state_update`. Verify passes `disable_state_update=True` and that `T.copy` is dead-code-eliminated. The scheduler later copies `ibuf[cache_slot, k_accepted]` into the live pool slot.

**Gating** host-side primary (A5). Decay `exp2(g·1.442695)` per step (raw g).

**`D=12` caveat.** Chunk-12 verify needs `D=12`, above the decode spine's `1..8` design point — re-derive `nreg` (§9 Gate 3) and the `M=1` gemm behavior at `D=12` with a compiled measurement.

**Varlen prologue (real kernel change).** SGLang passes `cu_seqlens = query_start_loc [N+1]` with `B=1` flattened; the CTA→request map (`bb` from a flattened token layout) is a **different prologue** than the dense `bb = bbh//H` (`fused_fwd.py:92`) — derive `bb` / per-request token ranges device-side (`kkt_solve.py:245-247`), no `.item()`. Re-verify the capture-safe index math.

**Perf (honest, conditional).** At `T=12`, server `N·H` saturating: bandwidth-bound on per-token state writes (bf16: 32 KB gather + 12·32 KB writes ≈ 416 KB/head; FLOPs hidden). V1 **beats** V2 here because the chunk kernel emits only 64-granular state (`NT=⌈12/64⌉=1`) so V2 must re-pay the same 12-step serial scan in a second pass plus chunk setup. Single-request (`N=1,H=16`): latency-bound, mitigated by CUDA-graph replay. Report per regime and per pool-dtype (falsifiable).

---

## 4. Verify V2 — chunk-fused `o` + honest per-token-state extraction

Built to benchmark; ships only if it wins total latency (§5).

**What the chunk kernel actually gives.** The WY/UT chunk algorithm yields state **only at chunk boundaries** (`ref_gdr.py` appends `last_state` once per chunk; `fused_fwd.py` writes state at the entry copy `:190` and the single `transpose_A` gemm `:204`). For `T=12`, `NT=⌈12/64⌉=1` ⇒ a **single** (initial) state, **not** per-token. Per-token states are **not** a free byproduct.

**The honest extraction mechanism + cost (the central V2 reality).** The post-loop extraction **cannot** reuse `vn_shared` as `v_new`: `vn_shared` holds `V' = g_rev·(Ag@W)` (`fused_fwd.py:283-286`) — the decay-corrected, A-projected **chunk** operand, **not** the per-token `v_new = β·(v − k·S_running)`. The true `v_new` depends on `w·S_running` (`w = A·k_beta`, WY-decoded), never materialized per-token. So extraction is a **genuinely new recurrent inner loop**: carry raw `v, k, β, g` (and per-token decay ratios `exp2(g_cumsum_t − g_cumsum_{t-1})`, also not resident) into a dedicated scan that, seeded from `S_entry`, recomputes `v_new_t` and runs 12 serial rank-1 updates, writing `S` after each token. **This scan IS V1's serial critical path, re-paid.** It needs an extra fp32 `S_scratch[DK,block_DV]` in the most register-pressured warpgroup (`CONSUMER_S_NREG=160`) → likely a **hard spill**, needing a dropped `nreg` or a dedicated scan warpgroup; and it must run **after** the `o` pipeline (`vn_shared` single-buffered, overwritten), serializing and killing the `o`/scan overlap that was V2's only hope.

**Cost summary at `T=12`.** V2 = (faster intra-chunk `o` via wgmma — the 1.8–2.16× probe, real, preserved) + (`kkt_solve` 64×64 inverse, **52/64 rows wasted** at T=12, an extra capturable launch) + (chunk cumsum) + (gate fusion across **three** surfaces: cumsum input, `kkt_solve` β load `:99`, `fused_fwd` `Ag` `:332`) + (**the same 12-step serial scan as V1**, now burdened with re-deriving `v_new`). State I/O is a **wash** vs V1 (same 12·32 KB writes). So the per-token-state requirement **erases V2's structural advantage**: V2 wins on `o`-latency only.

**Layout/gating fixes carried from A:** `state_v_first=True` for pool **and** ibuf; gate `g = -exp(A_log)·softplus(a+dt_bias)` **base-e** (any `exp2(A_log·1.442695)` form is **wrong** — the `1.442695/exp2` factor belongs only to the per-step decay application, `fused_fwd.py:236`); host-side l2norm for the MVP (note: this means the V2 entry is **not** fully in-kernel-l2norm capture-safe — scope it explicitly). The verify wrapper must **bypass the varlen branch** of `fused_gdr_fwd` entirely (`:569` calls `prepare_chunk_offsets`); for `T≤64` single-chunk, `chunk_offsets = arange(N+1)` / `chunk_indices = [[n,0]]` are static, precomputed pre-capture.

---

## 5. Benchmark-to-decide (V1 vs V2 at T=12)

Winner decided by **total verify-step latency (`o` + per-token states)**, not `o` alone.

- **Shapes:** `H=32, Hg=16, K=V=128, T=12`, single chunk. Sweep `N (=requests) ∈ {1, 8, 32, 64, 128, 256}` to cross the knee (`TARGET=⌊132·0.7⌋=92` on H100; `N·H` crosses 92 around `N=3`). `q,k` bf16 `[1, N·12, 16, 128]`; `v,o` bf16 `[1, N·12, 32, 128]`; `a,b` bf16 `[1, N·12, 32]`; `A_log,dt_bias` `[32]`; pool `[num_slots, 32, 128, 128]` V-major; `intermediate_states_buffer [num_cache_slots, 12, 32, 128, 128]`; indices int32. Run **both** bf16 and fp32 pools (the dominant per-token-write term doubles under fp32 and can flip the result).
- **Metrics:** (1) total wall time (median of 1000 iters, `flash_qla.utils.profile`); (2) achieved HBM GB/s vs ~3.35 TB/s, split state-read / per-token-write / io; (3) **V2 only:** separately time chunk-`o` vs the extraction scan and **measure overlap vs serialization**; (4) V2 `kkt_solve` launch+exec overhead (wasted 52/64 rows).
- **Baselines:** FLA `fused_sigmoid_gating_delta_rule_update` (the SGLang Triton verify kernel; natively emits per-token intermediates + supports `disable_state_update`) — both engines match its `o` and per-token states to 0.02 rel and target ≥ its SM90 perf. Also baseline V1 vs "12 separate `D=1` decode calls" (fusion win) and vs per-token-states-OFF (isolate the req #6 cost).
- **Decision rule:** ship **V1** unless V2 shows **≥15% total-latency win** at the deployment's dominant `N·H` **AND** its extraction scan **provably overlaps** the `o` pipeline **AND** passes the per-head-distinct-gate bit-identity test. **Pre-req to even running:** `M=1` gemm must compile (else fold M-pad-to-16 overhead into V1's measured time — do not assume hidden).

---

## 6. SGLang integration contract (verify entry, SM90, T=12 linear chain)

All tensors caller-preallocated; `B=1` outer dim, `N` requests flattened.

| tensor | shape | dtype | notes |
|---|---|---|---|
| `q`, `k` | `[1, N·T, 16, 128]` | bf16 | Hg=16 |
| `v`, `o` (returned) | `[1, N·T, 32, 128]` | bf16 | H=32; `o` written per token |
| `a`, `b` | `[1, N·T, 32]` | bf16 | raw gate inputs (in-kernel path); host path passes pre-activated `g`,`β` |
| `A_log`, `dt_bias` | `[32]` | fp32 | unused in host-gating baseline |
| `pool` (`ssm_states`) | `[num_slots, 32, 128, 128]` | SSM dtype (bf16) | **V-major** (`state_v_first=True`), K stride-1 |
| `state_indices` (`cache_indices`) | `[N]` | int32 | slot/request; `<0 ⇒ skip` gather AND scatter |
| `intermediate_states_buffer` | `[num_slots+1, 12, 32, 128, 128]` | SSM dtype | single-layer slice of `[num_layers, num_slots+1, draft_token_num=12, HV, V, K]`; **same V-major layout as pool**; `num_cache_slots = num_slots+1`; `cache_steps = 12` exact (non-adaptive) |
| `intermediate_state_indices` | `[N]` | int32 | destination slot into ibuf; **dense `arange`, never `-1`** — the ibuf write is gated by the **pool-slot mask** (`state_indices[b] ≥ 0`), not this index |
| `cu_seqlens` (`query_start_loc`) | `[N+1]` | int32 | ragged; in-kernel load only |

**Capture rules:** zero host sync (never `prepare_chunk_offsets`/`prepare_chunk_indices`; for V2 precompute trivial single-chunk offsets pre-capture and bypass the varlen branch); zero `torch.empty`; static shapes (`block_DV` from the MPC ladder, `T=12` fixed); the PyTorch gating/l2norm runs **inside** capture (capture-safe elementwise — A5), not in the TileLang kernel and **not** lifted out of the graph.

**No-commit:** `disable_state_update=True` ⇒ final pool scatter dead-code-eliminated; only `o` + ibuf produced. Decode follow-on sets it False and commits.

**Validation gate:** per-head-distinct-gate bit-identity vs FLA `fused_sigmoid_gating_delta_rule_update` (`o` + per-token states) — `K==V==128` makes a wrong `state_v_first` silent in equal-dim tests.

---

## 7. API signatures

**Graph-safe low-level entry (V1):**
```python
fused_recurrent_gdr_verify_fwd(
    q, k, v,                       # bf16 (raw gate path: + a, b bf16 [1,N·T,32]; A_log, dt_bias fp32 [32])
    pool,                          # [num_slots, 32, 128, 128] V-major; SSM dtype
    state_indices,                 # int32 [N]; -1 (PAD_SLOT_ID) => skip; slot 0 valid
    o,                             # CALLER-PREALLOC bf16 [1, N·T, 32, 128]
    intermediate_states_buffer,    # CALLER-PREALLOC [num_slots+1, 12, 32, 128, 128] V-major (single-layer slice)
    intermediate_state_indices,    # int32 [N]; dense arange (never -1); ibuf write gated by state_indices>=0
    cu_seqlens,                    # int32 [N+1]; per-CTA loop bound; in-kernel load only
    scale=None,                    # default 128**-0.5
    disable_state_update=True,     # NO-COMMIT (verify)
    use_qk_l2norm_in_kernel=False, # False in primary path (host l2norm)
    state_v_first=True,            # SGLang interop default
    allow_neg_eigval=False,        # decision; flag exposed
) -> o                              # pool written in-place ONLY if not disable_state_update
```
**JIT factory keys:** `tilelang_fused_recurrent_gdr_verify(H, Hg, DK=128, DV=128, scale, *_dtype, use_initial_state, disable_state_update, store_intermediate, is_varlen, use_qk_l2norm_in_kernel, state_v_first, allow_neg_eigval, fuse_gating, head_batch=False, group_size=1, block_DV=128, threads=256)`. `N`, `num_tokens` are `T.dynamic`; `D` fixed per capture. Kernel name ends `*_kernel_kernel` (CLAUDE.md).

**High-level wrapper** (drop-in for SGLang's DFlash `target_verify`; capture-safe PyTorch gating + dispatch, no allocation in the captured path):
```python
recurrent_gated_delta_rule_verify(
    A_log, a, dt_bias, q, k, v, b, ssm_states, cache_indices, query_start_loc,
    intermediate_states_buffer, intermediate_state_indices, cache_steps,
    retrieve_parent_token=None,    # accepted-and-IGNORED: DFlash is width-1 (TOPK=1); tree tensors are zeros
    scale=None, use_qk_l2norm_in_kernel=True, disable_state_update=True)
# asserts K==V==128, H%Hg==0, q.dtype!=fp32; derives state_v_first from ssm_states.stride()/shape;
# PyTorch l2norm+gating (primary, capture-safe — inside the graph, not the TileLang kernel);
# host block_DV via MPC ladder; no autograd. Confirm arg order vs DFlash's actual call (§10.6).
```
**V2 entry (only if benchmark-selected):** `tilelang_fused_chunk_gdr_verify_fwd(..., commit_final_state=False, store_intermediate=True, state_v_first=True, fuse_gate, ...)` with an explicit per-token extraction scan sub-kernel carrying raw `v,k,β,g`; static precomputed `chunk_offsets=arange(N+1)`/`chunk_indices=[[n,0]]`; bypasses `prepare_chunk_offsets`.

---

## 8. Test plan

- **Reference:** FLA `fused_sigmoid_gating_delta_rule_update` (matches the SGLang verify kernel; emits per-token states + supports `disable_state_update`). Both engines match `o` + per-token states to **0.02 rel**.
- **Bit-identity (hard layout gate):** per-head-**distinct**-gate test (each head a different `g`,`β`) vs the reference for `state_v_first=True` — equal-dim (`K==V==128`) tests are numerically blind to a layout transpose, so distinct-gate is mandatory.
- **Sweeps:** `T∈{1,3,8,12}`; ragged `cu_seqlens` (mixed accepted lengths e.g. `[1,5,12]`, compare only `o[:,:L_b]` and `ibuf[slot,:L_b]`); GQA `Hg=16/Hv=32` (verify head `h` reads k-head `h//2`); `g=0` SWA heads; bf16 **and** fp32 pool; `state_indices`/`intermediate_state_indices` with `<0` skip slots (assert untouched).
- **Negative controls:** cumsum'd g must break; pre-update read must break; `mod` GQA must break; a `[K,V]` (wrong-major) pool must break the distinct-gate test; garbage in skipped (`<0`) slots must leave committed outputs bit-unchanged.
- **Graph-safety test:** capture the verify call under `torch.cuda.graph`, replay, assert no allocation/sync errors and identical output.
- **Feasibility smoke tests:** §10 gates (especially `M=1` gemm, `T.serial(L)`, `state_v_first` transpose store) before full sign-off.

---

## 9. Open feasibility gates (prototype on the Hopper box, ordered by gating power)

1. **`M=1` gemm_v1** (decode-spec §11.A) — **root gate**. Every repo gemm is `M=64`. Compile-test `M=1` first; fallback M-pad-to-16 (quantify the 64× wgmma-row waste; at T=12 BW-bound it may hide — measure).
2. **single-role `T.serial(L)` with runtime per-CTA `L`** (§11.B) — grounded only in warp-spec `prepare_h.py:166`; confirm single-role. Static fallback: `T.serial(T)` + `if t<L` predicate + host zero-fill of `g`(decay=1)/`β`(rank-1=0) for `t≥L`.
3. **`nreg` at `threads=256, block_DV=128, D=12`** (§11.C) — re-measure spill via `set_max_nreg` at the D=12 verify point.
4. **In-kernel gating primitives** — `hasattr(T,'log2'/'rsqrt'/'log'/'exp')` + an SM90 lower test. If absent, in-kernel raw-gate fusion (req #5) and in-kernel l2norm are **infeasible** ⇒ host-side path mandatory (already primary). `sigmoid`/decay-`exp2` are pure-`exp2`.
5. **Two-vector outer-product rank-1 FMA** (§11.H) — only needed for head-batched SMEM-state; core uses the grounded `transpose_A` gemm (`:204`).
6. **`state_v_first` transpose store** — whether `T.copy` from a `[DK,block_DV]` fragment into a `[V,K]` slice emits a strided TMA store without an SMEM transpose stage. Validate on a **non-square** probe or byte-compare vs FLA (never the `128==128` test).
7. **(V2 only) extraction-scan budget** — the extra fp32 `S_scratch[DK,block_DV]` in the `CONSUMER_S_NREG=160` warpgroup likely spills; confirm a dedicated scan warpgroup or dropped `nreg` before claiming V2 viable.

---

## 10. Deployment facts (resolved against the live fork) + residual confirms

**RESOLVED (build to these):**
1. **`SGLANG_MAMBA_SSM_DTYPE` = bf16** — confirmed (the pool is bf16). Kernel stays dtype-agnostic; the fp32-pool perf note is informational.
2. **`intermediate_states_buffer = [num_layers, num_slots+1, draft_token_num=12, HV, dim, dim]`** (`memory_pool.py:353-357`). The kernel gets the **single-layer slice** `[num_slots+1, 12, HV, V, K]`; **`num_cache_slots = num_slots+1`** (the `+1`); `cache_steps = 12` exact (non-adaptive). V-major confirmed from strides (the `memory_pool.py:353` docstring says `[…,HV,K,V]` but the strides are V-major — trust strides). Wrapper still asserts `cache_steps ≥ T`.
3. **`allow_neg_eigval = False`** — confirmed (no config key; `fused_gdn_gating` is plain `sigmoid(b)`, no ×2). Flag exposed, defaults False.
4. **Slot sentinel `PAD_SLOT_ID = -1`** — confirmed. Guard with `slot >= 0` (`fused_sigmoid_gating_recurrent.py:98/131/204/230`); slot 0 is valid (not vLLM's 0/NULL). Only `state_indices` is `-1`-padded; `intermediate_state_indices` is dense `arange` (gate ibuf writes on the pool mask — A1/§3).
6. **DFlash = linear width-1** (`SPECULATIVE_EAGLE_TOPK=1`; `dflash_worker_v2` passes empty `topk_p`/`topk_index`; `retrieve_*` tree tensors are zeros, populated only when `topk>1`, `hybrid_linear_attn_backend.py:487-489`). The wrapper **accepts-and-ignores** `retrieve_parent_token` (the kernel still receives it, trivial for width-1).

**RESIDUAL (confirm before final ship):**
5. **FlashInfer numerics** — for SM90/Hopper the Triton fp32-accum path runs; confirm whether bit-matching the FlashInfer SM100 bf16-state adapter is a requirement.
6b. **DFlash entry parity** — reconcile `recurrent_gated_delta_rule_verify`'s exact arg order / buffer names against DFlash's live GDN verify call so it's drop-in.

---

## 11. Implementation order

1. **Prove the gates** §9.1 (`M=1` gemm) and §9.2 (`T.serial(L)`) on the Hopper box — these gate everything.
2. **Core `gs=1` decode kernel** (decode-spec §3–§6) + validate.
3. **Infra A** (A1 paging, A2 bf16+fp32-accum, A3 graph-safe entry, A4 `state_v_first`, A5 host gating) as keys/hooks on the kernel + a graph-safe wrapper.
4. **Verify V1** (per-token intermediate writes + no-commit + varlen `cu_seqlens` prologue + `D=12`), validated by the bit-identity + graph-safety tests vs FLA.
5. **Verify V2** (chunk-`o` + the honest extraction scan) — only after §9.7 budget check.
6. **Benchmark-to-decide** (§5) → ship the winner; the other stays documented.
7. **In-kernel gating** fast-follow if §9.4 passes.
8. **(Optional) standalone decode (C)** — later. (Tree-structured propagation is **out of scope** — not needed; the linear-chain design doesn't preclude adding it later.)
