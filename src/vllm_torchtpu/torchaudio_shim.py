# SPDX-License-Identifier: Apache-2.0
"""Minimal ``torchaudio`` stand-in for vLLM-Omni on images without torchaudio.

vLLM-Omni imports torchaudio at module import in its OpenAI API server
(``entrypoints/openai/audio_utils_mixin.py:28``) and lazily at runtime in the
MiniMax-H3 audio VAE (``diffusion/models/minimax_h3/vae.py:1145``). The shim
provides exactly the APIs those paths use:

* ``torchaudio.functional.resample`` and ``torchaudio.transforms.Resample``:
  real windowed-sinc (Hann) resampling in pure torch with torchaudio's
  default parameters; identity only when the rates are equal.
* ``torchaudio.transforms.Spectrogram`` / ``TimeStretch`` /
  ``InverseSpectrogram`` (API-server speed adjustment) and
  ``torchaudio.functional.melscale_fbanks`` (imported at module import by
  ``vllm_omni/utils/audio.py:8``): importable, raise ``NotImplementedError``
  when used.
* ``torchaudio.load``: soundfile-backed decode returning
  ``(waveform[C, T] float32, sample_rate)``; used by MiniMax-H3
  ``reference_video.load_audio_file`` for Ref2VA audio references. Raises
  ``RuntimeError`` on input soundfile cannot decode (e.g. mp4/aac) so callers'
  ffmpeg fallbacks still run.

The resampler is adapted from torchaudio's ``_get_sinc_resample_kernel`` /
``_apply_sinc_resample_kernel`` (BSD-2-Clause; notice above
``_sinc_resample_kernel``).

This module runs at the ``import`` stage of the patch registry, before
vLLM, torch or TorchTPU are imported. Its module-level imports are standard
library only; ``install_torchaudio_shim()`` may import torchaudio (and
therefore torch) when a torchaudio wheel is present.
"""

import importlib.util
import logging
import math
import sys
import types
from importlib.machinery import ModuleSpec
from typing import Any

_logger = logging.getLogger("vllm." + __name__)

_SHIM_MARKER = "_vllm_torchtpu_shim"


def _module_available(name: str) -> bool:
    if name in sys.modules:
        return sys.modules[name] is not None
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _module_importable(name: str) -> bool:
    """Import ``name``; on failure drop its partial ``sys.modules`` entries.

    A torchaudio wheel built for another torch/CUDA is found by ``find_spec``
    but raises at import (e.g. ``OSError: libcudart.so.13``).
    """
    if not _module_available(name):
        return False
    try:
        importlib.import_module(name)
    except (ImportError, OSError):
        for loaded in [m for m in sys.modules if m == name or m.startswith(name + ".")]:
            del sys.modules[loaded]
        return False
    return True


# _sinc_resample_kernel and resample are adapted from torchaudio
# torchaudio/functional/functional.py (_get_sinc_resample_kernel and
# _apply_sinc_resample_kernel), distributed under the BSD 2-Clause License:
#
# Copyright (c) 2017 Facebook Inc. (Soumith Chintala),
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
def _sinc_resample_kernel(
    orig_freq: int,
    new_freq: int,
    lowpass_filter_width: int,
    rolloff: float,
    dtype: Any,
) -> tuple[Any, int]:
    """Hann-windowed sinc kernels, one per output phase (gcd-reduced rates)."""
    import torch

    base_freq = min(orig_freq, new_freq) * rolloff
    width = math.ceil(lowpass_filter_width * orig_freq / base_freq)
    taps = torch.arange(-width, width + orig_freq, dtype=dtype)[None, None] / orig_freq
    phases = torch.arange(0, -new_freq, -1, dtype=dtype)[:, None, None] / new_freq
    t = ((phases + taps) * base_freq).clamp(-lowpass_filter_width, lowpass_filter_width)
    window = torch.cos(t * math.pi / lowpass_filter_width / 2) ** 2
    t = t * math.pi
    nonzero_t = torch.where(t == 0, torch.ones_like(t), t)
    sinc = torch.where(t == 0, torch.ones_like(t), torch.sin(nonzero_t) / nonzero_t)
    return sinc * window * (base_freq / orig_freq), width


def resample(
    waveform: Any,
    orig_freq: int,
    new_freq: int,
    lowpass_filter_width: int = 6,
    rolloff: float = 0.99,
    resampling_method: str = "sinc_interp_hann",
    beta: float | None = None,
) -> Any:
    """Resample the last dimension of ``waveform`` to ``new_freq``."""
    if orig_freq <= 0 or new_freq <= 0:
        raise ValueError("Original frequency and desired frequency should be positive")
    if orig_freq == new_freq:
        return waveform
    if resampling_method != "sinc_interp_hann":
        raise NotImplementedError(
            f"torchaudio shim only implements resampling_method='sinc_interp_hann', "
            f"got {resampling_method!r}; install torchaudio for other methods."
        )
    import torch

    gcd = math.gcd(int(orig_freq), int(new_freq))
    orig = int(orig_freq) // gcd
    new = int(new_freq) // gcd
    # Kernels are built on the host (float64 is unavailable on TPU).
    kernel, width = _sinc_resample_kernel(
        orig, new, lowpass_filter_width, rolloff, waveform.dtype
    )
    kernel = kernel.to(waveform.device)
    shape = waveform.shape
    flat = waveform.reshape(-1, shape[-1])
    length = flat.shape[-1]
    padded = torch.nn.functional.pad(flat, (width, width + orig))
    out = torch.nn.functional.conv1d(padded[:, None], kernel, stride=orig)
    out = out.transpose(1, 2).reshape(flat.shape[0], -1)
    target_length = math.ceil(new * length / orig)
    return out[..., :target_length].reshape(*shape[:-1], -1)


class Resample:
    """``torchaudio.transforms.Resample`` (construct, then call on a waveform)."""

    def __init__(
        self,
        orig_freq: int = 16000,
        new_freq: int = 16000,
        resampling_method: str = "sinc_interp_hann",
        lowpass_filter_width: int = 6,
        rolloff: float = 0.99,
        beta: float | None = None,
        *,
        dtype: Any = None,
    ) -> None:
        self.orig_freq = orig_freq
        self.new_freq = new_freq
        self.resampling_method = resampling_method
        self.lowpass_filter_width = lowpass_filter_width
        self.rolloff = rolloff
        self.beta = beta

    def forward(self, waveform: Any) -> Any:
        return resample(
            waveform,
            self.orig_freq,
            self.new_freq,
            self.lowpass_filter_width,
            self.rolloff,
            self.resampling_method,
            self.beta,
        )

    __call__ = forward


def _unsupported(api: str) -> NotImplementedError:
    return NotImplementedError(
        f"torchaudio.{api} is unavailable: torchaudio is not installed and "
        "vllm_torchtpu only provides a resampling shim. Install torchaudio "
        "to use this feature."
    )


class _UnsupportedTransform:
    _api = ""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise _unsupported(self._api)


class Spectrogram(_UnsupportedTransform):
    _api = "transforms.Spectrogram"


class TimeStretch(_UnsupportedTransform):
    _api = "transforms.TimeStretch"


class InverseSpectrogram(_UnsupportedTransform):
    _api = "transforms.InverseSpectrogram"


def melscale_fbanks(*args: Any, **kwargs: Any) -> Any:
    raise _unsupported("functional.melscale_fbanks")


def load(
    uri: Any,
    frame_offset: int = 0,
    num_frames: int = -1,
    normalize: bool = True,
    channels_first: bool = True,
    format: str | None = None,
    buffer_size: int = 4096,
    backend: str | None = None,
) -> tuple[Any, int]:
    """``torchaudio.load`` via soundfile: ``(waveform, sample_rate)``.

    vLLM-Omni's MiniMax-H3 ``load_audio_file`` falls back to soundfile/ffmpeg
    only on ``ImportError``/``RuntimeError``/``OSError``, so a missing
    soundfile or a failed read is raised as ``RuntimeError``. Without this
    function the shim raised ``AttributeError`` and Ref2VA requests with
    audio references failed.
    """
    if not normalize:
        raise _unsupported("load(normalize=False)")
    try:
        import soundfile
    except ImportError as exc:
        raise RuntimeError(
            "torchaudio shim load() requires soundfile; install torchaudio or soundfile"
        ) from exc
    import torch

    kwargs: dict[str, Any] = {}
    if format is not None:
        kwargs["format"] = format.upper()
    try:
        data, sample_rate = soundfile.read(
            uri,
            start=frame_offset,
            frames=num_frames,
            dtype="float32",
            always_2d=True,
            **kwargs,
        )
    except Exception as exc:
        raise RuntimeError(f"torchaudio shim failed to load {uri!r}: {exc}") from exc
    waveform = torch.from_numpy(data)
    if channels_first:
        waveform = waveform.t().contiguous()
    return waveform, int(sample_rate)


def _new_module(name: str, attrs: dict[str, Any], *, package: bool) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__spec__ = ModuleSpec(name, None, is_package=package)
    if package:
        module.__path__ = []
    module.__dict__.update(attrs)
    setattr(module, _SHIM_MARKER, True)
    return module


def install_torchaudio_shim() -> None:
    # -------------------------------------------------------------------------
    # Upstream torch_tpu Root Cause Tracking:
    # Upstream: vLLM-Omni's API server imports torchaudio at module import
    # (audio_utils_mixin.py:28), and the MiniMax-H3 audio VAE resamples
    # conditioning audio with it (vae.py:1145-1154).
    #
    # Problem on TPU: torchaudio is not a vllm-torchtpu/torch_tpu
    # dependency, so TPU images may ship without it, or with a wheel that
    # fails to load (e.g. OSError: libcudart.so.13).
    #
    # Fix: install the shim modules, only when vllm_omni is present and
    # torchaudio cannot be imported.
    #
    # REMOVAL / FIX CONDITION:
    # Remove once the TPU image ships a torchaudio wheel matching the pinned
    # torch, or vLLM-Omni imports torchaudio lazily and only on the code paths
    # that need it.
    # -------------------------------------------------------------------------
    if not _module_available("vllm_omni") or _module_importable("torchaudio"):
        return
    functional = _new_module(
        "torchaudio.functional",
        {"resample": resample, "melscale_fbanks": melscale_fbanks},
        package=False,
    )
    transforms = _new_module(
        "torchaudio.transforms",
        {
            "Resample": Resample,
            "Spectrogram": Spectrogram,
            "TimeStretch": TimeStretch,
            "InverseSpectrogram": InverseSpectrogram,
        },
        package=False,
    )
    root = _new_module(
        "torchaudio",
        {
            "__version__": "0.0.0+vllm_torchtpu_shim",
            "functional": functional,
            "transforms": transforms,
            "load": load,
        },
        package=True,
    )
    sys.modules["torchaudio.functional"] = functional
    sys.modules["torchaudio.transforms"] = transforms
    sys.modules["torchaudio"] = root
    _logger.info("Applied TPU patch: torchaudio shim (torchaudio is not importable).")
