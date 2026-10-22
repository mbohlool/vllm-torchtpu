# Sol sparse attention on TPU v7x: answer to pmotgi/torchtpu-diffusion-serving#2

This folder answers [pmotgi/torchtpu-diffusion-serving#2](https://github.com/pmotgi/torchtpu-diffusion-serving/issues/2) with a Pallas TPU kernel for Sol sparse attention (`sol_pallas.py`) and measurements on the issue's 8 real MiniMax-H3 attention inputs.

## Repositories

- **`vllm-torchtpu`** (`vllm-omni` + `vllm-torchtpu`): MiniMax-H3 TPU implementation on branch [mbohlool/vllm-torchtpu @ omni-minimax-h3](https://github.com/mbohlool/vllm-torchtpu/tree/omni-minimax-h3). The Sol Pallas kernel is in [this folder](https://github.com/mbohlool/vllm-torchtpu/tree/omni-minimax-h3-sol-pallas/examples/omni/minimax_h3/sol_attention) on branch `omni-minimax-h3-sol-pallas`.
- **`torchtpu-diffusion-serving` (`pmotgi`)**: standalone `torch-tpu` server at [pmotgi/torchtpu-diffusion-serving](https://github.com/pmotgi/torchtpu-diffusion-serving). [Issue #2](https://github.com/pmotgi/torchtpu-diffusion-serving/issues/2) contains the Sol JAX reference script `bench_tpu_sol_attention.py` and describes the `i2v_attention_inputs.safetensors` inputs.

## The questions in Issue #2

> we'd like to understand what are the achievable speedups we can get when using Sparse Attention mechanisms on TPUv7 given that they are important on our Video Diffusion pipelines.

> So our goal is to understand how "close" to 60% of the `FLOP bound` column can we get

For reference, the issue notes that an optimized Sol kernel on B300 reaches about 60% of the `FLOP bound`. Over all 8 samples the `FLOP bound` column is **3.55x**, so 60% of it is a **2.13x** speedup over dense attention.

## Short answer

- **Today:** the Sol Pallas kernel runs at **0.70x** of dense Splash attention (318.2 ms vs 223.7 ms over 8 samples, all 8 within the issue's error limits). That is **19.8% of the `FLOP bound`**, and 3.3x faster than the Issue #2 JAX reference.
- **Ceiling on TPU v7x:** with the query tiles TPU needs for efficiency (512–1024 rows), the best possible speedup, even at 100% hardware efficiency, is **1.85x (52% of the `FLOP bound`) at 512 rows** and **1.44x (41%) at 1024 rows**. So over all 8 samples, **60% of the `FLOP bound` (2.13x) is above what 512-row tiles can reach.** Only tiles of 256 rows or fewer have a higher ceiling (2.41x), and they would need to run at about 88% of it.
- **What is achievable:** beating dense needs a faster kernel at the same tile size. The kernel currently reaches 38% of the 1.85x ceiling; an untuned setting already reached 0.94x of dense on the largest sample (step 3).

## Step 1: the ceiling on TPU v7x tiles

The `FLOP bound` column in `bench_tpu_sol_attention.py` assumes each 64-row query block loads only its own routed 64-token KV blocks: 26.6% of the dense KV blocks over all 8 samples. The TPU matrix unit needs larger query tiles to stay efficient, and a tile must load the **union** of the KV blocks routed by any of its 64-row query blocks. Larger tiles therefore load more blocks.

Over all 8 samples (FLOP-weighted):

| Query rows per tile | 64 (exact) | 128 | 256 | 512 | 1024 |
|---|---:|---:|---:|---:|---:|
| Union density (share of dense KV blocks loaded) | 26.6% | 31.8% | 39.9% | 52.5% | 67.7% |
| Best possible speedup vs dense | 3.55x | 3.00x | 2.41x | **1.85x** | **1.44x** |
| Share of the `FLOP bound` | 100% | 85% | 68% | **52%** | **41%** |

Per sample, the 512-row ceiling is 49%–69% of the `FLOP bound`. The four smaller samples (sample3–sample6) are at 60%–69%; the four large ones, which dominate the total time, are at 49%–54%.

## Step 2: what the Pallas kernel reaches

`sol_pallas.py` is a block-sparse + centroid Sol kernel. Routing runs in XLA; the kernel does one online softmax over the centroids of unrouted blocks and the routed KV blocks, which it gathers in groups of 16 blocks per step. Timed on one TensorCore of a TPU v7x `2x2x1` host, over all 8 samples of the Issue #2 inputs, including routing:

| Implementation | Time | vs Issue #2 JAX ref | vs dense |
|---|---:|---:|---:|
| MaxDiffusion Splash dense (current `vllm-torchtpu` kernel, 4864x1024 tiles) | 223.7 ms | – | 1.00x |
| Issue #2 JAX ref (Sol) | 1044.9 ms | 1.00x | 0.21x |
| **Sol Pallas** (512-row tiles) | **318.2 ms** (kernel only: 310.4 ms) | **3.28x faster** | **0.70x** |

Per sample, 512-row tiles:

| Sample | Tokens | Dense (ms) | Sol Pallas (ms) | Speedup | `FLOP bound` | % of `FLOP bound` | Tile-union ceiling | % of ceiling | Error (limit) | p99 (limit) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| sample0 | 77,574 | 42.75 | 67.74 | 0.631x | 3.20x | 19.7% | 1.68x | 37.7% | 0.0604 (0.0635) | 0.4375 (0.4595) |
| sample1 | 34,419 | 9.87 | 15.34 | 0.644x | 3.10x | 20.8% | 1.59x | 40.5% | 0.1354 (0.1423) | 0.6317 (0.6634) |
| sample2 | 103,947 | 77.83 | 114.71 | 0.678x | 3.59x | 18.9% | 1.76x | 38.5% | 0.1009 (0.1060) | 0.5203 (0.5465) |
| sample3 | 24,139 | 4.40 | 7.82 | 0.562x | 2.49x | 22.6% | 1.55x | 36.2% | 0.1115 (0.1172) | 0.7167 (0.7526) |
| sample4 | 12,218 | 1.41 | 2.16 | 0.652x | 2.35x | 27.7% | 1.63x | 39.9% | 0.1138 (0.1196) | 0.6353 (0.6673) |
| sample5 | 26,486 | 5.69 | 9.07 | 0.628x | 2.55x | 24.6% | 1.56x | 40.1% | 0.0554 (0.0583) | 0.3997 (0.4198) |
| sample6 | 16,716 | 2.47 | 4.30 | 0.574x | 2.34x | 24.5% | 1.41x | 40.6% | 0.0817 (0.0859) | 0.5896 (0.6192) |
| sample7 | 106,065 | 79.30 | 97.09 | 0.817x | 4.07x | 20.1% | 2.18x | 37.5% | 0.0980 (0.1030) | 0.4732 (0.4970) |
| **all** | | **223.7** | **318.2** | **0.703x** | **3.55x** | **19.8%** | **1.85x** | **38.0%** | mean 0.0946 | |

- **Quality:** all 8 samples are within the issue's mean and p99 error limits against dense attention. The kernel output differs from the Issue #2 JAX reference by only 1.4e-4 (mean error), so it has the same quality as the reference.
- **1024-row tiles:** 319.1 ms (0.701x), which is 19.8% of the `FLOP bound` and **48.6% of the 1.44x ceiling** (47.2%–52.1% per sample).
- **Reproducible:** an independent re-run of the 512-row configuration measured 321.6 ms.
- The timings were taken before two small robustness fixes in `sol_pallas.py` (stricter input validation, and a padding edge case that only occurs when `tokens % 64 == 0`, which none of the 8 samples hit). The fixed kernel gives identical results in CPU interpret-mode correctness checks but has not been re-timed on TPU.

## Step 3: why it reaches only 38%–49% of the ceiling, and what would beat dense

Likely causes (not yet profiled):

- **Small gathers.** Each routed KV block is a 64-row, 16 KB copy from HBM. Splash streams contiguous 1024x128 (256 KB) KV tiles that keep the matrix unit busy.
- **Dense centroid pass.** Every query tile scores the centroids of all blocks.
- **Less query reuse.** A 512-row query tile reuses each loaded KV block less than Splash's 4864-row tiles.

A sweep of blocks gathered per step shows headroom: with 32 blocks per step (512-row tiles), the largest sample (sample7, 106,065 tokens) ran in 84.7 ms vs 79.5 ms for dense, **0.94x**, and 1024-row tiles reached 56.5% of the ceiling on sample1. These sweep settings have not been quality-checked or run on all 8 samples yet.

To beat dense, the kernel needs fewer and larger gathers (merge adjacent blocks, deeper buffering), a cheaper centroid pass, and smaller tiles without losing efficiency. Getting near 60% of the `FLOP bound` also needs tiles of 256 rows or fewer, because larger tiles cap it lower (step 1).

## Step 4: how to run

`sol_pallas.py` is self-contained: it needs only `jax` (tested with `jax`/`jaxlib` 0.11.2, `libtpu` 0.0.48.1, Python 3.12). `sol_route` in it is copied from the Issue #2 script.

### Call the kernel directly

```python
from functools import partial

import jax
from sol_pallas import sol_attention_pallas

# q, k, v: (tokens, heads, 128) bfloat16, after RMS norm and RoPE.
# sink_tokens: number of leading context rows (text, keyframe and audio) that Sol keeps exact.
attn = jax.jit(partial(sol_attention_pallas, sink_tokens=sink_tokens, tau=1.0))
out = attn(q, k, v)  # (tokens, heads, 128) float32
```

The defaults (`Tq=512` query rows per tile, `G=16` KV blocks per step, `Bc=256`) are the configuration timed above. `Tq` must be 128, 256, 512 or 1024. Pass `interpret=True` to run the kernel on CPU for correctness checks.

### Reproduce the benchmark with the Issue #2 script

1. Save `bench_tpu_sol_attention.py` from the body of [Issue #2](https://github.com/pmotgi/torchtpu-diffusion-serving/issues/2) next to `sol_pallas.py`, together with the `i2v_attention_inputs.safetensors` inputs file the issue describes.
2. In `bench_tpu_sol_attention.py`, swap the JAX reference for the kernel:

   ```python
   from sol_pallas import sol_attention_pallas

   # in main(), replace the `sparse = jax.jit(partial(sol_attention, ...))` line with:
   sparse = jax.jit(partial(sol_attention_pallas, sink_tokens=sink_tokens, tau=c.tau))
   ```

3. Run it on a TPU v7x host:

   ```bash
   uv run bench_tpu_sol_attention.py --inputs i2v_attention_inputs.safetensors
   ```

   The script times dense Splash attention against the Sol kernel on all 8 samples and fails if any sample exceeds its mean or p99 error limit. Its `speedup` and `FLOP bound` columns give the "% of `FLOP bound`" above.

## End-to-end MiniMax-H3 serving on TPU v7x

For context, the full MiniMax-H3 pipeline (real weights, TPU v7x `2x2x1`, 4 chips / 8 TensorCores, dense attention) on `vllm-torchtpu` compared with the numbers in the `pmotgi/torchtpu-diffusion-serving` README. 124 frames at 24 fps; each number is the mean of warm requests 2 and 3.

| Shape | Setup | s/step | Denoise (s) | End-to-end (s) | Speedup vs. pmotgi TP8 (step / E2E) |
|---|---|---:|---:|---:|---:|
| 512x288, 20 steps | pmotgi README TP8 | 0.217 | 4.12 | 8.40 | – |
| | vllm-torchtpu TP8 | 0.200 | 3.99 | 6.99 | 1.09x / 1.20x |
| | **vllm-torchtpu TP2xU4** | **0.124** | **2.48** | **5.49** | **1.75x / 1.53x** |
| 1344x768, 20 steps | pmotgi README TP8 | 2.066 | 39.25 | 52.05 | – |
| | vllm-torchtpu TP8 | 1.533 | 30.67 | 34.31 | 1.35x / 1.52x |
| | **vllm-torchtpu TP2xU4** | **1.131** | **22.63** | **26.29** | **1.83x / 1.98x** |
| 1344x768, 50 steps | pmotgi README TP8 | 2.266 | 123.09 | 135.90 | – |
| | vllm-torchtpu TP8 | 1.530 | 76.49 | 80.08 | 1.48x / 1.70x |
| | **vllm-torchtpu TP2xU4** | **1.124** | **56.22** | **59.86** | **2.02x / 2.27x** |

- The `pmotgi` stack was not re-run. Its numbers come from the `torchtpu-diffusion-serving` README, measured on the same hardware type.
- The `pmotgi` server syncs after every step and `vllm-torchtpu` does not. At 1344x768 the progress-bar step rate matches the measured step time, so the speedup there is real. At 512x288, part of the small TP8 gain may come from the different sync pattern.
