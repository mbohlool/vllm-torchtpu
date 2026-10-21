# MiniMax-H3 on TPU v7x with vLLM-Omni

MiniMax-H3 (`MiniMaxAI/MiniMax-H3`) is a joint video and audio diffusion pipeline served by [vLLM-Omni](https://github.com/vllm-project/vllm-omni). This recipe runs the released `MiniMaxAI/MiniMax-H3` checkpoint across every MiniMax-H3 pipeline (text-to-video+audio, first/last-frame-to-video+audio and reference-to-video+audio, offline and online) on one TPU v7x host (`tpu7x-8`, 8 logical TPU devices, `TP=8` or `TP=2` x Ulysses `4`) with `vllm-torchtpu` and `vllm-omni`.

---

## 1. Quick Start

On a TPU v7x machine (`tpu7x-8`: 4 chips, 8 logical devices) with about `140 GB` of free disk:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh && source $HOME/.local/bin/env
git clone -b omni-minimax-h3 https://github.com/mbohlool/vllm-torchtpu.git && cd vllm-torchtpu
uv venv --python 3.12 && source .venv/bin/activate
uv pip install --prerelease=allow ".[omni]" --group vllm
python examples/omni/minimax_h3/run_all_pipelines.py --only 1_T2VA
```

The last command writes `minimax_h3_outputs/1_T2VA.mp4` (a 4 s `480x480` video with audio, 50 steps) of a puppy running through a meadow. Add `--prompt "..."` to generate your own scene; describe the subject, setting, lighting and sound, because a vague prompt such as `"A test prompt"` tends to produce a text title card. The first run downloads the checkpoint (~`138 GB`, Section 2), so it takes much longer than later runs.

With `uv`, the `omni` extra installs `vllm-omni` at commit `e1d77a3` (pinned in `[tool.uv.sources]` in `pyproject.toml`), and `--group vllm` builds `vllm` `v0.31.0` from source. `torchaudio` is not required: `vllm-torchtpu` installs a minimal `torchaudio` shim when `torchaudio` cannot be imported.

---

## Summary of Validated Pipelines

| Pipeline | Task | Conditioning | Entry Point |
| :--- | :--- | :--- | :--- |
| `1_T2VA` | `t2va` | text only | `run_all_pipelines.py` (offline, `AsyncOmni`) |
| `2_FL2VA_0` | `fl2va` | 1 image at frame `0` | `run_all_pipelines.py` |
| `3_FL2VA_last` | `fl2va` | 1 image at frame `-1` | `run_all_pipelines.py` |
| `4_FL2VA_0_last` | `fl2va` | 2 images at frames `0, -1` | `run_all_pipelines.py` |
| `5_Ref2VA_image` | `ref2va` | reference image | `run_all_pipelines.py` |
| `6_Ref2VA_image_audio` | `ref2va` | reference image + audio | `run_all_pipelines.py` |
| `7_Ref2VA_video` | `ref2va` | reference video | `run_all_pipelines.py` |
| `8_online` | `t2va` | text only, `audio_mode=native` | `server_and_curl.py` (`vllm serve --omni`, `POST /v1/videos/sync`) |

By default the example scripts configure `TP=8`, text-encoder TP `8`, VAE patch parallel `8` (`tile` mode, VAE tiling on), `DP=1`, eager mode. For higher-resolution sequences (`480x480` and `768x768`), `TP=2` x Ulysses `4` (`tensor_parallel_size=2, ulysses_degree=4`) reduces DiT communication volume and improves steady-state step latency.

---

## 2. Checkpoint (`MiniMaxAI/MiniMax-H3`)

On the first run, `model="MiniMaxAI/MiniMax-H3"` (or `vllm serve MiniMaxAI/MiniMax-H3`) downloads `FL2VA/**`, `Ref2VA/model_index.json` and `Ref2VA/transformer/**` (~`138 GB`, the same set for every task) from Hugging Face into the Hugging Face cache (`~/.cache/huggingface`; set `HF_HOME` to use a larger disk). The download runs during stage initialization, so both example scripts raise `stage_init_timeout` / `init_timeout` to `1800 s`. On a slow link, download the same files before the first run:

```bash
hf download MiniMaxAI/MiniMax-H3 --include "FL2VA/**" "Ref2VA/model_index.json" "Ref2VA/transformer/**"
```

Later runs reuse the cached snapshot and download nothing again (vLLM-Omni still asks the Hub for the current revision unless `HF_HUB_OFFLINE=1` is set). You can also pass a local snapshot directory as the model.

> [!NOTE]
> The released `MiniMaxAI/MiniMax-H3` audio and video VAEs ship as custom remote code inside the checkpoint (`FL2VA/audio_vae` and `FL2VA/video_vae`), so the scripts pass `trust_remote_code=True` / `--trust-remote-code`.

---

## 3. Conditioning Inputs and `ffmpeg`

The image, audio and video conditioned pipelines (`2_FL2VA_0` to `7_Ref2VA_video`) read small test media (`256x256`, 24 fps, 4 s). Write them to `minimax_h3_inputs/` (`--output-dir` to change):

```bash
python examples/omni/minimax_h3/generate_inputs.py
```

This writes `image_color_bars.png`, `image_gradient.png`, `image_checkerboard.png`, `video_moving_gradient.mp4` and `audio_sine_sweep.wav`.

`7_Ref2VA_video` also needs `ffmpeg` and `ffprobe` on `PATH` (vLLM-Omni's Ref2VA `reference_video.py` calls both):

```bash
sudo apt-get update && sudo apt-get install -y ffmpeg
```

---

## 4. Run the Pipelines

Run from the repository root with the venv active. The defaults are `--steps 50 --size 480 --duration 4` (full 50-step generation, seed `42`, 24 fps); pass for example `--size 768 --duration 5` for a larger video. For a quick check of the setup pass `--steps 2 --size 256 --duration 4`: two steps do not denoise the video, so its frames stay mostly white noise. Outputs go to `minimax_h3_outputs/` (`--output-dir`); inputs are read from `minimax_h3_inputs/` (`--inputs-dir`). `6_Ref2VA_image_audio` and `7_Ref2VA_video` pass file paths to the pipeline (`audio_path` and `[video_path]`), not raw bytes.

### Offline pipelines (`AsyncOmni`)

```bash
# Text to video + audio, 50 steps at 768x768, 5 s.
python examples/omni/minimax_h3/run_all_pipelines.py --only 1_T2VA --steps 50 --size 768 --duration 5

# All 7 offline pipelines, 2 steps at 256x256, 4 s, in one engine.
python examples/omni/minimax_h3/run_all_pipelines.py --steps 2 --size 256 --duration 4

# One pipeline at a time: 1_T2VA, 2_FL2VA_0, 3_FL2VA_last, 4_FL2VA_0_last,
# 5_Ref2VA_image, 6_Ref2VA_image_audio or 7_Ref2VA_video.
python examples/omni/minimax_h3/run_all_pipelines.py --only 7_Ref2VA_video
```

The script builds the engine with:

```python
AsyncOmni(
    model="MiniMaxAI/MiniMax-H3",
    trust_remote_code=True,
    tensor_parallel_size=8,
    text_encoder_tp_size=8,
    vae_patch_parallel_size=8,
    data_parallel_size=1,
    num_gpus=8,
    enforce_eager=True,
    vae_parallel_mode="tile",
    vae_use_tiling=True,
)
```

### Online pipeline (`8_online`)

```bash
python examples/omni/minimax_h3/server_and_curl.py
```

The script starts the server, waits until `GET /v1/models` answers, sends one request and stops the server. To do the same by hand:

```bash
vllm serve MiniMaxAI/MiniMax-H3 --omni --trust-remote-code \
  --num-gpus 8 \
  --tensor-parallel-size 8 \
  --text-encoder-tp-size 8 \
  --vae-patch-parallel-size 8 \
  --vae-parallel-mode tile \
  --vae-use-tiling \
  --enforce-eager \
  --stage-init-timeout 1800 \
  --init-timeout 1800 \
  --port 8000
```

```bash
curl -X POST http://localhost:8000/v1/videos/sync \
  --data-urlencode prompt="A test prompt" \
  --data-urlencode size=480x480 \
  --data-urlencode seconds=4 \
  --data-urlencode num_inference_steps=50 \
  --data-urlencode extra_params='{"audio_mode": "native", "task": "t2va", "aspect_ratio": "1:1"}' \
  -o 8_online.mp4
```

`t2va` needs an explicit `aspect_ratio`.

---

## 5. Expected Outputs

For each pipeline `<name>` the scripts write to `minimax_h3_outputs/`:

- `<name>.mp4`: H.264 video (`yuv420p`, 24 fps, requested size) muxed with a 32 kHz stereo AAC audio track. For `8_online` it is the body of the `/v1/videos/sync` response; if the server returns JSON, the script decodes the base64 video from it and verifies the MP4 `ftyp` header before saving.
- `<name>_frame0.png` and `<name>_mid.png`: first and middle decoded frames (offline pipelines). `8_online` writes `8_online_frame0.png`.
- `run_status.json`: per-pipeline `status`, `seconds`, `error`, the `ffmpeg` stream lines of the MP4 and the PNG names.

---

## 6. Results (`MiniMaxAI/MiniMax-H3` on TPU v7x, `tpu7x-8`)

All numbers were measured with the released `MiniMaxAI/MiniMax-H3` checkpoint on one TPU v7x host (`tpu7x-8`, 8 logical TPU devices), `vllm-omni` commit `e1d77a3`, `t2va`, `seed=42`, `flow_shift=12.0`, `audio_flow_shift=3.0`.

### Example scripts on this branch

`run_all_pipelines.py` and `server_and_curl.py` as in Section 4, `TP=8`. Times are the request times in `run_status.json` (engine start-up and checkpoint loading not included); a pipeline whose shapes are already compiled in the process takes about 1.5 s at 2 steps.

| Pipeline | Configuration | Request time |
| :--- | :--- | :--- |
| `1_T2VA` | `256x256`, 4 s, 2 steps | `42.98 s` |
| `2_FL2VA_0` | `256x256`, 4 s, 2 steps | `44.35 s` |
| `3_FL2VA_last` | `256x256`, 4 s, 2 steps | `1.50 s` |
| `4_FL2VA_0_last` | `256x256`, 4 s, 2 steps | `42.34 s` |
| `5_Ref2VA_image` | `256x256`, 4 s, 2 steps | `1.49 s` |
| `6_Ref2VA_image_audio` | `256x256`, 4 s, 2 steps | `73.43 s` |
| `7_Ref2VA_video` | `256x256`, 4 s, 2 steps | `79.72 s` |
| `1_T2VA` | `480x480`, 4 s, 50 steps (first request) | `77.35 s` |
| `1_T2VA` | `768x768`, 5 s, 2 steps (first request) | `71.19 s` |
| `8_online` | `256x256`, 4 s, 2 steps | `44.30 s` |

### DiT step time and warm request time

Measured with a benchmark harness around `AsyncOmni` that sends the same request three times to one engine (twice for the 50-step rows) and times the denoise loop separately. `TP2xU4` is `tensor_parallel_size=2, ulysses_degree=4` in the engine arguments; the example scripts use `TP=8`. "Before" is this branch without its performance commits (up to "Video VAE device context through torch.accelerator").

8 steps, warm DiT `s/step` (requests 2 and 3) and second-request time:

| Resolution & duration | Tokens (`used` / `padded`) | Before, `TP8` | This branch, `TP8` | This branch, `TP2xU4` |
| :--- | :--- | :--- | :--- | :--- |
| `256x256`, 4 s | `2,408` / `2,432` | `0.497 s/step`, `6.61 s` | `0.093 s/step`, `1.48 s` | — |
| `480x480`, 4 s | `7,560` / `7,616` | `0.438 s/step`, `4.76 s` | `0.237 s/step`, `3.23 s` | `0.159 s/step`, `2.65 s` |
| `768x768`, 5 s | `21,730` / `21,760` | `1.257 s/step`, `11.90 s` | `0.750 s/step`, `7.85 s` | `0.530 s/step`, `6.06 s` |
| `768x768`, 10 s | `42,286` / `42,304` | `2.903 s/step`, `27.04 s` | `1.789 s/step`, `17.99 s` | `1.355 s/step`, `14.59 s` |

50 steps, second-request time:

| Resolution & duration | Before, `TP8` | This branch, `TP8` | This branch, `TP2xU4` |
| :--- | :--- | :--- | :--- |
| `480x480`, 4 s | `23.66 s` | `12.55 s` (`0.224 s/step`) | `8.33 s` |
| `768x768`, 5 s | `64.85 s` | `38.81 s` (`0.740 s/step`) | `27.33 s` |

On this branch requests 2 and 3 return bit-identical frames. Before, the second request at `256x256` still recompiled its first two steps (its third request took `2.78 s`). `TP2xU4` and `TP8` frames differ by rounding (8 steps: `28.4` / `25.4` / `25.5 dB` PSNR at `480x480x4s` / `768x768x5s` / `768x768x10s`).

---

## 7. TPU Patches Applied

`vllm-torchtpu` applies these patches automatically; no user action is needed. Each one is a no-op when vLLM-Omni or the MiniMax-H3 pipeline is not loaded.

| Patch | Location | Why |
| :--- | :--- | :--- |
| Released checkpoint layout detection | `src/vllm_torchtpu/omni/minimax_h3.py` (`patch_minimax_h3_checkpoint_modular`) | `MiniMaxAI/MiniMax-H3` ships both `modular_model_index.json` at the root and `FL2VA/` + `Ref2VA/` subdirectories without `fastvideo_inference.json`; the patch routes local snapshots without `fastvideo_inference.json` to the `FL2VA` / `Ref2VA` Diffusers partition. |
| Video VAE equal-shape tile gather | `src/vllm_torchtpu/omni/minimax_h3.py` (`patch_video_vae_tile_gather`) | Pads short ranks with zeros to `ceil(num_tiles / sp_size)` in the tile's float dtype and runs one equal-shape `dist.all_gather` between device sync fences instead of an int64 shape gather and unequal-shape `all_gather`. |
| TPU-safe tensor broadcasts | `src/vllm_torchtpu/omni/minimax_h3.py` (`patch_pipeline_tensor_broadcasts`) | Broadcasts shape headers and payloads as whole host-staged tensors over the coordinator's `cpu_group` (or fenced int32 device tensors). |
| TPU-safe object collectives | `src/vllm_torchtpu/omni/minimax_h3.py` (`patch_pipeline_object_collectives`) | Routes `broadcast_object_list` / `all_gather_object` over the gloo `cpu_group` when available, or a host-built length header plus int32 payload. |
| Indexed modulation reference | `src/vllm_torchtpu/omni/minimax_h3.py` (`patch_indexed_modulation`) | Dispatches the DiT AdaLN indexed-modulation ops to PyTorch reference implementations on devices without Triton. |
| Video VAE device context | `src/vllm_torchtpu/omni/minimax_h3.py` (`patch_video_vae_device_module`) | Replaces `TpuDeviceModule.device` with `torch.accelerator.device_index` during keyframe-encode reseeding. |
| AdaLN cache host-sync bypass | `src/vllm_torchtpu/omni/minimax_h3.py` (`patch_adaln_projection_cache`) | Keeps `ExactProjectionCache` (the AdaLN exact-projection memo) unarmed on TPU to avoid per-step host hash syncs and per-block TP hit-vote syncs. |
| Fixed-width denoise timestep table | `src/vllm_torchtpu/omni/minimax_h3.py` (`patch_denoise_branch_timestep_rows`) | Pads `unique_timesteps` to `H3_UNIQUE_TIMESTEP_ROWS = 4` on TPU so step 0 and later steps share one static DiT compilation shape. |
| Key-padding Pallas FlashAttention (`kv_len`) | `src/vllm_torchtpu/omni/attention.py` (`TpuFlashAttentionImpl`) | Routes MiniMax-H3 `[real, pad]` key-masked calls to `pallas::diffusion_flash_attention_kv_len` without host syncs or recompilation. |
| Sync-free denoise loop & deferred finite checks | `src/vllm_torchtpu/omni/minimax_h3.py` (`patch_denoise_loop_rope_table`, `patch_scheduler_finite_checks`) | Prebuilds the packed RoPE table once per loop, replaces `.nonzero()` gathers/scatters with static index selects/scatters, and defers finite checks to a single host sync at the end of the loop. |
| Fused per-head q/k RMSNorm + RoPE kernel | `src/vllm_torchtpu/omni/minimax_h3.py` (`patch_attention_fused_qk_norm_rope`) | Fuses per-head q/k RMSNorm and partial 96-of-128 NeoX RoPE into one Pallas kernel (`pallas::minimax_h3_qk_norm_rope`). |
| Full-span DiT embed under Ulysses SP | `src/vllm_torchtpu/omni/minimax_h3.py` (`patch_ulysses_full_span_embed`) | Keeps `_rope_local_span` on the full `(0, seq_len)` span on TPU when `ulysses_degree > 1` so no rank feeds a zero-row slice into `ColumnParallelLinear` all-gather. |
| Diffusion async output stream device | `src/vllm_torchtpu/omni/patches.py` (`_patch_worker_proc_async_output_device`) | Binds the worker D2H side stream to local device index `0` on TPU. |
| Diffusion IPC tensor packing | `src/vllm_torchtpu/omni/patches.py` (`_safe_pack_tensor_if_large`) | Moves tensors to CPU before IPC serialization and forwards `created=` and `enable_registered_shm=`. |
| `torchaudio` shim | `src/vllm_torchtpu/torchaudio_shim.py` | Provides `torchaudio.load` (via `soundfile`) and resampling on TPU hosts without `torchaudio`. |

---

## 8. Known Limitations

- Execution uses eager mode (`--enforce-eager`).
- The disaggregated deployments (`minimax_h3_disaggregated*.yaml`) do not start on TPU: their prompt-encoder stage needs an Omni connector model runner, which the TPU worker does not provide (`ValueError: Worker 'TPUWorker' does not provide an Omni connector model runner`). Use the single-stage commands above.
- With the `torchaudio` shim, API-server speed adjustment (`Spectrogram` / `TimeStretch` / `InverseSpectrogram`) raises `NotImplementedError`. Install `torchaudio` if speed adjustment is required.
