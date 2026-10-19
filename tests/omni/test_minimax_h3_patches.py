# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the MiniMax-H3 / vLLM-Omni TPU patches.

These tests never import vLLM-Omni or MiniMax-H3 code.
"""

import sys
import types
from types import SimpleNamespace

import pytest
import torch

from vllm_torchtpu import torchaudio_shim
from vllm_torchtpu.omni import patches as omni_patches

pytestmark = pytest.mark.omni

# -----------------------------------------------------------------------------
# (c) torchaudio shim
# -----------------------------------------------------------------------------


@pytest.fixture
def torchaudio_modules():
    names = ("torchaudio", "torchaudio.functional", "torchaudio.transforms")
    saved = {name: sys.modules[name] for name in names if name in sys.modules}
    yield
    for name in names:
        sys.modules.pop(name, None)
    sys.modules.update(saved)


def _install_with(monkeypatch, *, torchaudio, vllm_omni=True):
    monkeypatch.setitem(sys.modules, "torchaudio", torchaudio)
    monkeypatch.setitem(
        sys.modules, "vllm_omni", types.ModuleType("vllm_omni") if vllm_omni else None
    )
    torchaudio_shim.install_torchaudio_shim()
    return sys.modules["torchaudio"]


def test_torchaudio_shim_only_when_missing(monkeypatch, torchaudio_modules):
    real = types.ModuleType("torchaudio")
    assert _install_with(monkeypatch, torchaudio=real) is real
    assert _install_with(monkeypatch, torchaudio=None, vllm_omni=False) is None

    shim = _install_with(monkeypatch, torchaudio=None)
    assert getattr(shim, "_vllm_torchtpu_shim", False)
    # Idempotent: a second call keeps the installed shim.
    torchaudio_shim.install_torchaudio_shim()
    assert sys.modules["torchaudio"] is shim

    import torchaudio
    import torchaudio.functional as functional
    from torchaudio.functional import melscale_fbanks
    from torchaudio.transforms import InverseSpectrogram, Spectrogram, TimeStretch

    assert torchaudio is shim
    assert functional.resample is torchaudio_shim.resample
    with pytest.raises(NotImplementedError, match="melscale_fbanks"):
        melscale_fbanks(201, 0.0, 8000.0, 80, 16000)
    for cls in (Spectrogram, TimeStretch, InverseSpectrogram):
        with pytest.raises(NotImplementedError, match=cls.__name__):
            cls(n_fft=768)


def test_torchaudio_shim_version(monkeypatch, torchaudio_modules):
    version = pytest.importorskip("packaging.version")
    shim = _install_with(monkeypatch, torchaudio=None)
    assert getattr(shim, "_vllm_torchtpu_shim", False)
    assert isinstance(shim.__version__, str)
    assert version.Version(shim.__version__).release == (0, 0, 0)


def test_torchaudio_shim_when_import_fails(monkeypatch, torchaudio_modules):
    import importlib
    import importlib.machinery

    for name in [n for n in sys.modules if n.split(".")[0] == "torchaudio"]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.setitem(sys.modules, "vllm_omni", types.ModuleType("vllm_omni"))
    find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name, *args: (
            importlib.machinery.ModuleSpec(name, None)
            if name == "torchaudio"
            else find_spec(name, *args)
        ),
    )
    import_module = importlib.import_module

    def failing_import(name, package=None):
        if name != "torchaudio":
            return import_module(name, package)
        # A wheel built for CUDA registers part of the package, then fails.
        sys.modules["torchaudio"] = types.ModuleType("torchaudio")
        sys.modules["torchaudio._extension"] = types.ModuleType("_extension")
        raise OSError("libcudart.so.13: cannot open shared object file")

    monkeypatch.setattr(importlib, "import_module", failing_import)
    torchaudio_shim.install_torchaudio_shim()

    shim = sys.modules["torchaudio"]
    assert getattr(shim, "_vllm_torchtpu_shim", False)
    assert "torchaudio._extension" not in sys.modules
    assert sys.modules["torchaudio.functional"].resample is torchaudio_shim.resample


def test_torchaudio_shim_resample():
    waveform = torch.randn(2, 1000)
    assert torchaudio_shim.resample(waveform, 16000, 16000) is waveform
    assert torchaudio_shim.Resample(24000, 24000)(waveform) is waveform

    for orig, new in ((16000, 24000), (48000, 16000), (44100, 48000)):
        out = torchaudio_shim.resample(waveform, orig, new)
        assert out.shape == (2, -(-1000 * new // orig))
        assert out.dtype == waveform.dtype
    torch.testing.assert_close(
        torchaudio_shim.Resample(16000, 24000)(waveform),
        torchaudio_shim.resample(waveform, 16000, 24000),
    )
    assert torchaudio_shim.resample(torch.randn(1000), 16000, 8000).shape == (500,)

    # A band-limited sine is reproduced away from the edges.
    orig, new, freq = 16000, 24000, 440.0
    t_in = torch.arange(4000, dtype=torch.float64) / orig
    t_out = torch.arange(6000, dtype=torch.float64) / new
    out = torchaudio_shim.resample(
        torch.sin(2 * torch.pi * freq * t_in).float(), orig, new
    )
    expected = torch.sin(2 * torch.pi * freq * t_out).float()
    torch.testing.assert_close(out[200:-200], expected[200:-200], atol=1e-2, rtol=0)

    with pytest.raises(NotImplementedError, match="sinc_interp_hann"):
        torchaudio_shim.resample(
            waveform, 16000, 8000, resampling_method="sinc_interp_kaiser"
        )


def test_torchaudio_shim_load(tmp_path, monkeypatch, torchaudio_modules):
    soundfile = pytest.importorskip("soundfile")
    import numpy as np

    rate, length = 32000, 8000
    tone = np.sin(2 * np.pi * 440.0 * np.arange(length) / rate)
    data = (tone[:, None] * np.array([[0.5, -0.25]])).astype(np.float32)
    path = tmp_path / "tone.wav"
    soundfile.write(path, data, rate, subtype="FLOAT")

    waveform, sample_rate = torchaudio_shim.load(str(path))
    assert sample_rate == rate
    assert waveform.dtype == torch.float32
    assert waveform.shape == (2, length)
    torch.testing.assert_close(waveform, torch.from_numpy(data).t())

    part, _ = torchaudio_shim.load(
        path, frame_offset=100, num_frames=50, channels_first=False
    )
    assert part.shape == (50, 2)
    torch.testing.assert_close(part, torch.from_numpy(data[100:150]))

    # Undecodable input raises RuntimeError so callers' ffmpeg fallbacks run.
    bogus = tmp_path / "clip.mp4"
    bogus.write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 64)
    with pytest.raises(RuntimeError, match="failed to load"):
        torchaudio_shim.load(str(bogus))
    with pytest.raises(NotImplementedError, match="normalize"):
        torchaudio_shim.load(str(path), normalize=False)

    shim = _install_with(monkeypatch, torchaudio=None)
    assert shim.load is torchaudio_shim.load
    import torchaudio

    assert torchaudio.load(str(path))[1] == rate


# -----------------------------------------------------------------------------
# (d) async output loop device
# -----------------------------------------------------------------------------


class _FakeWorkerProc:
    def __init__(self):
        self.gpu_id = 3
        self.seen_gpu_id = None
        self.pending = 2

    def _helper(self):
        return self

    def _async_output_loop(self):
        """Original loop."""
        self.seen_gpu_id = self.gpu_id
        self.pending = self.pending - 1
        assert self._helper() is self._helper()


@pytest.fixture
def fake_worker_cls():
    cls = type("FakeWorkerProc", (_FakeWorkerProc,), {})
    cls._async_output_loop = _FakeWorkerProc._async_output_loop
    return cls


@pytest.mark.parametrize(
    ("accelerator", "expected_gpu_id"),
    [(SimpleNamespace(type="tpu"), 0), (SimpleNamespace(type="cuda"), 3), (None, 3)],
)
def test_async_output_loop_device(
    monkeypatch, fake_worker_cls, accelerator, expected_gpu_id
):
    monkeypatch.setattr(torch.accelerator, "current_accelerator", lambda: accelerator)
    assert omni_patches._patch_async_output_loop_device(fake_worker_cls)
    patched = fake_worker_cls._async_output_loop
    assert not omni_patches._patch_async_output_loop_device(fake_worker_cls)
    assert fake_worker_cls._async_output_loop is patched
    assert patched.__doc__ == "Original loop."

    worker = fake_worker_cls()
    worker._async_output_loop()
    assert worker.seen_gpu_id == expected_gpu_id
    assert worker.gpu_id == 3
    assert worker.pending == 1  # writes reach the real worker
