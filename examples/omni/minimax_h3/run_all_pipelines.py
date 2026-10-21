"""Run the offline MiniMax-H3 pipelines (T2VA, FL2VA, Ref2VA) with AsyncOmni.

Each pipeline writes <name>.mp4, two PNG frames and an entry in
run_status.json under --output-dir (default: ./minimax_h3_outputs). Inputs
come from generate_inputs.py (--inputs-dir, default: ./minimax_h3_inputs).
Ref2VA video conditioning needs ffmpeg and ffprobe on PATH.

Usage, from the repository root inside the venv (defaults: --steps 50
--size 480 --duration 4):

    python examples/omni/minimax_h3/run_all_pipelines.py --only 1_T2VA

Pass --prompt "..." to generate your own scene. For a quick check of the
setup, pass --steps 2 --size 256. Two steps do not denoise the video, so its
frames stay mostly white noise.
"""

import argparse
import asyncio
import copy
import json
import os
import subprocess
import tempfile
import time
import traceback
import wave
from pathlib import Path

import imageio_ffmpeg
import numpy as np
from PIL import Image
from vllm_omni.entrypoints.async_omni import AsyncOmni

# A short, concrete scene with sound. A vague prompt such as "A test prompt"
# tends to produce a text title card instead of a scene.
T2VA_PROMPT = (
    "A golden retriever puppy runs through a sunlit meadow of wildflowers "
    "toward the camera, ears flapping, tall grass swaying in the breeze. "
    "Warm late-afternoon light, shallow depth of field, cinematic. "
    "Birds chirp and the puppy barks happily."
)


def engine_kwargs() -> dict:
    return {
        "model": "MiniMaxAI/MiniMax-H3",
        "trust_remote_code": True,
        "tensor_parallel_size": 8,
        "text_encoder_tp_size": 8,
        "vae_patch_parallel_size": 8,
        "data_parallel_size": 1,
        "num_gpus": 8,
        "enforce_eager": True,
        "vae_parallel_mode": "tile",
        "vae_use_tiling": True,
        # The first run downloads the checkpoint (~138 GB) during stage
        # initialization; the vLLM-Omni defaults (300 s / 600 s) are too short.
        "stage_init_timeout": 1800,
        "init_timeout": 1800,
    }


def get_sampling_params(
    engine,
    task: str,
    duration: int,
    steps: int,
    size: int,
    frame_indices: list | None = None,
):
    params = copy.deepcopy(engine.default_sampling_params_list)
    diff = params[0]
    diff.width = size
    diff.height = size
    diff.fps = 24
    diff.num_inference_steps = steps
    diff.seed = 42
    diff.extra_args = {
        "task": task,
        "duration": float(duration),
        "aspect_ratio": "1:1",
        "flow_shift": 12.0,
        "audio_flow_shift": 3.0,
    }
    if frame_indices is not None:
        diff.extra_args["frame_indices"] = frame_indices
    return params


def save_mp4(frames, audio, path):
    """Mux (T, H, W, 3) uint8 frames and stereo 32 kHz audio into an H.264/AAC mp4."""
    _, H, W, _ = frames.shape
    exe = imageio_ffmpeg.get_ffmpeg_exe()

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        wav_path = f.name
    with wave.open(wav_path, "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(32000)
        # The pipeline returns (1, 2, samples); WAV wants interleaved (samples, 2).
        if audio.ndim == 3:
            audio_flat = audio[0].transpose(1, 0)
        elif audio.ndim == 2:
            audio_flat = audio.transpose(1, 0)
        else:
            audio_flat = audio
        if audio_flat.dtype in (np.float32, np.float64):
            audio_flat = (np.clip(audio_flat, -1, 1) * 32767).astype(np.int16)
        wf.writeframes(audio_flat.tobytes())

    cmd = [
        exe,
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-s",
        f"{W}x{H}",
        "-pix_fmt",
        "rgb24",
        "-r",
        "24",
        "-i",
        "-",
        "-i",
        wav_path,
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        str(path),
    ]
    proc = subprocess.Popen(
        cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    stdout, stderr = proc.communicate(frames.tobytes())
    os.remove(wav_path)
    if proc.returncode != 0:
        print(f"ffmpeg failed with code {proc.returncode}")
        print(stderr.decode())


async def run_all(args):
    engine = AsyncOmni(**engine_kwargs())

    inputs_dir = Path(args.inputs_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Inputs load lazily so that 1_T2VA (text only) runs without them.
    def img(name):
        return Image.open(inputs_dir / name).convert("RGB")

    img1 = "image_color_bars.png"
    img2 = "image_gradient.png"

    # Ref2VA takes audio and video references as file paths; raw bytes are
    # rejected (vllm-omni encoder_processing.py, reference_video.py).
    video_path = str(inputs_dir / "video_moving_gradient.mp4")
    audio_path = str(inputs_dir / "audio_sine_sweep.wav")

    pipelines = [
        {
            "name": "1_T2VA",
            "task": "t2va",
            "prompt": T2VA_PROMPT,
            "mm_data": lambda: {},
        },
        {
            "name": "2_FL2VA_0",
            "task": "fl2va",
            "prompt": "A test prompt",
            "mm_data": lambda: {"image": img(img1)},
            "frames": [0],
        },
        {
            "name": "3_FL2VA_last",
            "task": "fl2va",
            "prompt": "A test prompt",
            "mm_data": lambda: {"image": img(img1)},
            "frames": [-1],
        },
        {
            "name": "4_FL2VA_0_last",
            "task": "fl2va",
            "prompt": "A test prompt",
            "mm_data": lambda: {"image": [img(img1), img(img2)]},
            "frames": [0, -1],
        },
        {
            "name": "5_Ref2VA_image",
            "task": "ref2va",
            "prompt": "A test prompt",
            "mm_data": lambda: {"image": img(img1)},
        },
        {
            "name": "6_Ref2VA_image_audio",
            "task": "ref2va",
            "prompt": "A test prompt",
            "mm_data": lambda: {"image": img(img1), "audio": audio_path},
        },
        {
            "name": "7_Ref2VA_video",
            "task": "ref2va",
            "prompt": "A test prompt",
            "mm_data": lambda: {"video": [video_path]},
        },
    ]

    only_names = [n.strip() for n in args.only.split(",")] if args.only else []

    # Merge into an existing run_status.json so separate --only runs add up.
    status_file = out_dir / "run_status.json"
    run_status = {}
    if status_file.exists():
        try:
            with open(status_file) as f:
                run_status = json.load(f)
        except Exception as e:
            print(f"Could not load existing status file: {e}")

    for pipe in pipelines:
        if only_names and pipe["name"] not in only_names:
            continue

        print(f"--- Running {pipe['name']} ---", flush=True)
        t0 = time.time()
        status_entry = {
            "status": "success",
            "seconds": 0.0,
            "error": None,
            "mp4": None,
            "png": [],
        }
        try:
            prompt_dict = {
                "prompt": args.prompt or pipe["prompt"],
                "multi_modal_data": pipe["mm_data"](),
            }

            async for output in engine.generate(
                prompt=prompt_dict,
                request_id=pipe["name"],
                sampling_params_list=get_sampling_params(
                    engine,
                    pipe["task"],
                    args.duration,
                    args.steps,
                    args.size,
                    pipe.get("frames"),
                ),
            ):
                if output.finished:
                    frames = np.asarray(output.images[0])  # (T, H, W, 3)
                    audio = np.asarray(output.multimodal_output["audio"])
                    mp4_path = out_dir / f"{pipe['name']}.mp4"
                    save_mp4(frames, audio, mp4_path)
                    print(f"Saved {mp4_path}", flush=True)

                    png0_path = out_dir / f"{pipe['name']}_frame0.png"
                    pngmid_path = out_dir / f"{pipe['name']}_mid.png"
                    Image.fromarray(frames[0]).save(png0_path)
                    Image.fromarray(frames[len(frames) // 2]).save(pngmid_path)

                    # Record ffmpeg's stream lines as a check of the mp4.
                    exe = imageio_ffmpeg.get_ffmpeg_exe()
                    proc = subprocess.run(
                        [exe, "-hide_banner", "-i", str(mp4_path)],
                        capture_output=True,
                        text=True,
                    )
                    stream_lines = [
                        line.strip()
                        for line in proc.stderr.split("\n")
                        if "Stream" in line
                    ]

                    status_entry["mp4"] = stream_lines
                    status_entry["png"] = [png0_path.name, pngmid_path.name]
        except Exception as e:
            traceback.print_exc()
            status_entry["status"] = "error"
            status_entry["error"] = str(e)

        status_entry["seconds"] = time.time() - t0
        run_status[pipe["name"]] = status_entry

        with open(status_file, "w") as f:
            json.dump(run_status, f, indent=2)

    engine.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only",
        type=str,
        default="",
        help="comma-separated pipeline names to run, e.g. 1_T2VA (default: all)",
    )
    parser.add_argument(
        "--prompt",
        default="",
        help="text prompt for every selected pipeline (default: a puppy scene "
        "for 1_T2VA, 'A test prompt' for the others)",
    )
    parser.add_argument("--steps", type=int, default=50, help="inference steps")
    parser.add_argument(
        "--size", type=int, default=480, help="output width and height in pixels"
    )
    parser.add_argument(
        "--duration", type=int, default=4, help="output length in seconds"
    )
    parser.add_argument(
        "--inputs-dir",
        default="minimax_h3_inputs",
        help="directory written by generate_inputs.py",
    )
    parser.add_argument(
        "--output-dir",
        default="minimax_h3_outputs",
        help="directory for the mp4s, PNG frames and run_status.json",
    )
    args = parser.parse_args()
    asyncio.run(run_all(args))
