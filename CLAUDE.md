# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

FlashQLA is a single-operation kernel library: fused, warp-specialized **TileLang** kernels for the **Gated Delta Rule (GDN) chunked prefill** forward and backward passes, targeting NVIDIA **Hopper (SM90)**. It is a drop-in faster alternative to the Flash-Linear-Attention (FLA) Triton kernels (2-3x forward, 2x backward). Everything in the repo exists to compute `chunk_gated_delta_rule`.

Hard environment requirements: SM90 or above, CUDA 12.8+, PyTorch 2.8+, Python 3.10+. Pinned deps: `tilelang==0.1.8`, `apache-tvm-ffi==0.1.9`.

## Commands

```bash
# Install (editable install matters: setup.py embeds the git SHA in the version)
pip install -v .

# Lint/format — pre-commit is formatting-only (ruff); real linting is CI's job
pre-commit run -a          # run on all files
pre-commit install         # install the git hook

# Correctness + speedup tests — require fla for the reference comparison
pip install flash_linear_attention==0.5.0
cd tests
python test_gdr.py --set develop                                   # quick smoke
python test_gdr.py --set varlen   --num-heads 32                   # variable-length batches
python test_gdr.py --set profile  --num-heads 32                   # latency sweep
python test_gdr.py --set product  --ref-dtype float32 --num-heads 32

# Static signature guard — CPU-only, no GPU/fla needed (parses source via ast)
python tests/test_function_signature.py

# Benchmark vs FLA Triton + FlashInfer
pip install flash_linear_attention==0.5.0 flashinfer-python==0.6.9
cd benchmark
python bench_gated_delta_rule.py
```

`test_gdr.py` is driven by a `--set <name>` preset that loads `tests/settings/<name>.csv` (develop / varlen / profile / product), with per-run overrides via flags: `--num-heads`/`--nvh`, `--nkh`, `--seqlen`, `--no-h0`, `--skip-bwd`, `--no-cp` (disable auto context-parallelism), `--swa-ratio`, `--data-dtype`, `--ref-dtype`, `--hide-acc`, `--hide-lat`, `--seed`. There is no single-test selector beyond editing the CSV; a "test" is one shape row. The accuracy check runs the QLA kernel **1000 times in a loop** asserting the result stays within 2% of the reference — this is a deliberate race/nondeterminism detector, not redundancy.

## Architecture

Three layers, top to bottom:

**1. Public API** — `flash_qla/__init__.py` re-exports three entry points:
- `chunk_gated_delta_rule` — the autograd-wrapped op most callers use. It is `@torch.compiler.disable`d and wraps `ChunkGatedDeltaRuleFunction` (a `torch.autograd.Function`).
- `chunk_gated_delta_rule_fwd` / `chunk_gated_delta_rule_bwd` — low-level functions that bypass autograd (what the tests and benchmark call directly). The forward returns intermediates (`g, A, o, h, final_state`) that the backward needs fed back in.

**2. Orchestration** — `flash_qla/ops/gated_delta_rule/chunk/__init__.py` sequences kernels into passes and enforces all the input invariants:
- Forward: `chunk_local_cumsum(g)` → `kkt_solve` (builds the `A` interaction matrix) → optional `intra_card_cp_preprocess` (auto context-parallelism) → `fused_gdr_fwd`.
- Backward: `fused_gdr_h` (recompute hidden states `h`) → `fused_gdr_bwd` → `group_reduce_vector` on `dq`/`dk` when GQA (`Hg < H`) → reverse `chunk_local_cumsum(dg)`.
- Invariants enforced here (changing them breaks the kernels): `head_dim_k == head_dim_v == 128`, `chunk_size == 64`, dtype is bf16/fp16 (**not** fp32), `num_v_heads % num_k_heads == 0` (GQA), `head_first=False` only, and batch size must be 1 when `cu_seqlens` is given.

**3. Kernels** — `flash_qla/ops/gated_delta_rule/chunk/hopper/`. Dispatch is hard-gated: both `chunk/__init__.py` and `cp_context.py` check `tilelang.contrib.nvcc.get_target_compute_version() == "9.0"` and **raise** otherwise. Kernel modules:
- `fused_fwd.py` (`fused_gdr_fwd`) — the main fused forward.
- `fused_bwd.py` (`fused_gdr_bwd`) — the main fused backward.
- `prepare_h.py` (`fused_gdr_h`) — recomputes hidden state `h`; used by both backward and CP preprocessing.
- `kkt_solve.py` (`kkt_solve`) — solves the lower-triangular `(I - tril(βKKᵀ))` system into `A`.
- `cp_fwd.py` (`get_warmup_chunks`, `correct_initial_states`) — context-parallelism helpers.

### Intra-card context parallelism (the distinctive part)

`cp_context.py` is the headline optimization. When `auto_cp=True` and `batch_size == 1`, it splits one long sequence into shorter CP sub-sequences to raise SM occupancy under TP / long-context / small-head-count regimes. Flow:
- `_calc_cp_seqs` picks `max_local_chunks` from a latency model (`L_cp* ∝ √(B·H·L_c/P)`, ×3 empirical factor, rounded to a power of 2; floored at 4 for pipelining) and decides `use_cp` based on whether `B·H` already saturates the SMs.
- It exploits the GDN gate's exponential decay: `get_warmup_chunks` finds how many preceding chunks each split needs to "warm up" a usable initial state (gate threshold `-10.0`); `fused_gdr_h` computes those carry states; `correct_initial_states` stitches them back.
- Output threads `cp_seq_map` (cp-batch → raw-batch index) and `raw_cu_seqlens` into the fused kernel via its `is_cp` flag, so the kernel writes final states to the correct raw-batch slots.

### Utilities

- `flash_qla/ops/utils/` — `chunk_local_cumsum` (`cumsum.py`), `group_reduce_vector` (`group_reduce.py`); both are TileLang kernels.
- `flash_qla/utils/` — `l2norm` (`math.py`); varlen packing helpers `pack`/`unpack`/`pad_and_reshape`/`fill_last_chunk_of_g` (`pack.py`); `profile` (`profiler.py`); `tensor_cache`/`prepare_chunk_indices`/`prepare_chunk_offsets` (`index.py`).

## Conventions and gotchas

- **Kernel definition pattern**: each `@tilelang.jit`-decorated function is a *factory* taking shape/dtype/flag parameters and returning a `@T.prim_func`. The plain-Python wrapper (e.g. `fused_gdr_fwd`) computes shapes/dtypes/flags, picks `block_DV` from the grid size vs SM count, calls the factory, then invokes the returned kernel. Every distinct parameter combination triggers a fresh JIT compile.
- **Tensor layouts** are fixed: `q`/`k` are `[B, T, Hg, 128]`, `v`/`o` are `[B, T, H, 128]`, `g`/`beta` are `[B, T, H]`, states are `[B, H, 128, 128]`. `Hg` = K/Q heads, `H` = V heads, `Hg ≤ H`.
- **Varlen mode**: `batch_size` must be 1, inputs are flattened/packed along `T`, and `cu_seqlens` marks sequence boundaries (see `pack`/`unpack`).
- **Kernel names are an API for the tests/benchmark**: `profile()` keys timings by kernel name, and `test_gdr.py`/`bench_gated_delta_rule.py` index specific names (e.g. `tilelang_fused_chunk_gdr_fwd_kernel_kernel`, `tilelang_kkt_solve_kernel_kernel`, `tilelang_prepare_h_kernel_kernel`, `tilelang_correct_h0_kernel_kernel`). Renaming a kernel silently breaks those lookups.
- **The fused kernels hard-code their thread geometry**: 512 threads split into one producer + three consumer warpgroups, with manual named-barrier `arrive_count`s (96/128/256/384/416) tied to that layout. These are not tunable knobs — they encode the warp-specialization schedule.
- **`tensor_cache`** (`index.py`) is an identity-based LRU (keyed on `id()`, size 256), used for CP/index helper tensors — it caches by object identity, not value.
- **`backward` gradient count is guarded statically**: `test_function_signature.py` parses the source with `ast` to assert `ChunkGatedDeltaRuleFunction.backward` returns exactly one gradient per non-`ctx` forward input (a real past bug — PR #10). Runs without a GPU. If you change `forward`'s signature, update `backward`'s return tuple in lockstep.
- Backward `dg` must be fp32 (asserted in the orchestration layer before the reverse cumsum).
