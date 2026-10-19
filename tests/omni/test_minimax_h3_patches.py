# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the MiniMax-H3 / vLLM-Omni TPU patches.

Distributed tests run a real 2-process gloo group. The pipeline under test is
a stand-in module exposing the attributes the patches replace, so these tests
never import vLLM-Omni or MiniMax-H3 code.
"""

import importlib.util
import sys
import types
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from vllm_torchtpu import torchaudio_shim
from vllm_torchtpu.omni import minimax_h3
from vllm_torchtpu.omni import patches as omni_patches

pytestmark = pytest.mark.omni

_CPU = torch.device("cpu")
_WORLD_SIZE = 2


# -----------------------------------------------------------------------------
# Helpers shared with the spawned ranks (module level so spawn can import them).
# -----------------------------------------------------------------------------


def _expected_tensors() -> list[torch.Tensor]:
    return [
        torch.arange(7, dtype=torch.long),
        (torch.arange(12, dtype=torch.float32) * 0.5).reshape(3, 4).to(torch.bfloat16),
        torch.arange(24, dtype=torch.long).reshape(2, 3, 4) - 5,
        torch.linspace(-1.0, 1.0, 120).reshape(2, 3, 4, 5).to(torch.bfloat16),
    ]


def _assert_same(actual: torch.Tensor, expected: torch.Tensor) -> None:
    assert actual.shape == expected.shape, (actual.shape, expected.shape)
    assert actual.dtype == expected.dtype, (actual.dtype, expected.dtype)
    assert torch.equal(actual, expected)


def _fake_pipeline_module(
    device_group, cpu_group, rank: int, world_size: int
) -> types.ModuleType:
    module = types.ModuleType("fake_pipeline_minimax_h3")
    module.dist = torch.distributed
    module._dit_rank_world = lambda: (device_group, rank, world_size)
    module.get_world_group = lambda: SimpleNamespace(
        device_group=device_group, cpu_group=cpu_group
    )

    def _broadcast_tensor(tensor, *, dtype, device):
        raise AssertionError("unpatched _broadcast_tensor")

    class MiniMaxH3Pipeline:
        def _encoder_group_broadcast_tensor(self, tensor, *, dtype, device):
            raise AssertionError("unpatched _encoder_group_broadcast_tensor")

        def _distribute_encode_inputs(self, ids, vision_kwargs):
            raise AssertionError("unpatched _distribute_encode_inputs")

    module._broadcast_tensor = _broadcast_tensor
    module.MiniMaxH3Pipeline = MiniMaxH3Pipeline
    return module


def _init_gloo(rank: int, world_size: int, init_file: str) -> None:
    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=world_size
    )


def _check_distribute_encode_inputs(pipeline, rank, device_group, cpu_group) -> None:
    instance = pipeline.MiniMaxH3Pipeline()
    instance.text_encoder_group = SimpleNamespace(
        world_size=_WORLD_SIZE,
        rank_in_group=rank,
        ranks=[0, 1],
        device_group=device_group,
        cpu_group=cpu_group,
    )
    instance.device = _CPU
    ids = torch.arange(11, dtype=torch.long)
    grid = torch.tensor([[1, 4, 6]], dtype=torch.long)
    pixels = torch.linspace(0.0, 1.0, 24 * 8).reshape(24, 8).to(torch.bfloat16)
    vision = {"pixel_values": pixels, "image_grid_thw": grid} if rank == 0 else {}
    out_ids = instance._distribute_encode_inputs(ids if rank == 0 else None, vision)
    _assert_same(out_ids, ids)
    assert sorted(vision) == ["image_grid_thw", "pixel_values"]
    _assert_same(vision["pixel_values"], pixels)
    _assert_same(vision["image_grid_thw"], grid)


def _metadata_worker(rank: int, world_size: int, init_file: str) -> None:
    _init_gloo(rank, world_size, init_file)
    try:
        device_group = dist.group.WORLD
        cpu_group = dist.new_group(list(range(world_size)), backend="gloo")
        # Core helper with the gloo cpu_group and with the device fallback.
        for metadata_group in (cpu_group, None):
            for expected in _expected_tensors():
                out = minimax_h3.broadcast_tensor(
                    expected if rank == 0 else None,
                    dtype=expected.dtype,
                    device=_CPU,
                    src=0,
                    is_src=rank == 0,
                    device_group=device_group,
                    cpu_group=metadata_group,
                    meta_len=minimax_h3.DIT_META_LEN,
                )
                _assert_same(out, expected)

        # Patched pipeline functions (DiT world group and encoder TP group).
        for metadata_group in (cpu_group, None):
            pipeline = _fake_pipeline_module(
                device_group, metadata_group, rank, world_size
            )
            assert minimax_h3.patch_pipeline_tensor_broadcasts(pipeline)
            for expected in _expected_tensors():
                out = pipeline._broadcast_tensor(
                    expected if rank == 0 else None,
                    dtype=expected.dtype,
                    device=_CPU,
                )
                _assert_same(out, expected)
            # dtype conversion on the way out, as upstream.
            out = pipeline._broadcast_tensor(
                torch.arange(6).reshape(2, 3) if rank == 0 else None,
                dtype=torch.bfloat16,
                device=_CPU,
            )
            _assert_same(out, torch.arange(6).reshape(2, 3).to(torch.bfloat16))
            _check_distribute_encode_inputs(
                pipeline, rank, device_group, metadata_group
            )
        dist.barrier()
    finally:
        dist.destroy_process_group()


def _object_worker(rank: int, world_size: int, init_file: str) -> None:
    _init_gloo(rank, world_size, init_file)
    try:
        orig_broadcast = dist.broadcast_object_list
        orig_gather = dist.all_gather_object
        device_group = dist.group.WORLD
        cpu_group = dist.new_group(list(range(world_size)), backend="gloo")
        pipeline = _fake_pipeline_module(device_group, cpu_group, rank, world_size)
        assert minimax_h3.patch_pipeline_object_collectives(pipeline)
        assert not minimax_h3.patch_pipeline_object_collectives(pipeline)

        # torch.distributed itself stays untouched.
        assert torch.distributed.broadcast_object_list is orig_broadcast
        assert torch.distributed.all_gather_object is orig_gather
        shim = pipeline.dist
        assert shim is not torch.distributed
        assert shim.get_rank is torch.distributed.get_rank
        assert shim.broadcast is torch.distributed.broadcast
        assert shim.cpu_group_for(device_group) is cpu_group
        assert shim.cpu_group_for(object()) is None

        expected = [
            {"type": "ValueError", "message": "boom", "status_code": 400},
            ({"num_frames": 9}, [torch.bfloat16, torch.long]),
        ]
        payload = list(expected) if rank == 0 else [None, None]
        shim.broadcast_object_list(payload, src=0, group=device_group)
        assert payload == expected

        gathered = [None] * world_size
        shim.all_gather_object(
            gathered, ("rank", rank, "x" * (rank * 100 + 3)), group=device_group
        )
        assert gathered == [("rank", r, "x" * (r * 100 + 3)) for r in range(world_size)]

        # Fixed-protocol fallback used when no gloo cpu_group exists.
        payload = list(expected) if rank == 0 else [None, None]
        minimax_h3.fallback_broadcast_object_list(
            payload, src=0, group=device_group, device=_CPU
        )
        assert payload == expected
        gathered = [None] * world_size
        minimax_h3.fallback_all_gather_object(
            gathered,
            {"rank": rank, "blob": b"\x01" * (rank * 7 + 1)},
            group=device_group,
            device=_CPU,
        )
        assert gathered == [
            {"rank": r, "blob": b"\x01" * (r * 7 + 1)} for r in range(world_size)
        ]
        dist.barrier()
    finally:
        dist.destroy_process_group()


def _spawn(worker, tmp_path) -> None:
    mp.spawn(
        worker,
        args=(_WORLD_SIZE, str(tmp_path / "gloo_init")),
        nprocs=_WORLD_SIZE,
        join=True,
    )


# -----------------------------------------------------------------------------
# (a) metadata broadcast, (b) object collectives
# -----------------------------------------------------------------------------


def test_metadata_broadcast_over_gloo(tmp_path):
    _spawn(_metadata_worker, tmp_path)


def test_object_collectives_shim_over_gloo(tmp_path):
    _spawn(_object_worker, tmp_path)


def test_single_rank_fast_paths_and_errors():
    pipeline = _fake_pipeline_module(None, None, 0, 1)
    minimax_h3.patch_pipeline_tensor_broadcasts(pipeline)
    out = pipeline._broadcast_tensor(torch.arange(3), dtype=torch.bfloat16, device=_CPU)
    _assert_same(out, torch.arange(3).to(torch.bfloat16))
    with pytest.raises(ValueError, match="source tensor is required"):
        pipeline._broadcast_tensor(None, dtype=torch.long, device=_CPU)

    instance = pipeline.MiniMaxH3Pipeline()
    instance.text_encoder_group = SimpleNamespace(world_size=1, rank_in_group=0)
    instance.device = _CPU
    _assert_same(
        instance._distribute_encode_inputs(torch.arange(4, dtype=torch.int32), {}),
        torch.arange(4, dtype=torch.long),
    )
    with pytest.raises(ValueError, match="encoder rank 0 must produce input ids"):
        instance._distribute_encode_inputs(None, {})

    pipeline = _fake_pipeline_module(None, None, 0, 2)
    minimax_h3.patch_pipeline_tensor_broadcasts(pipeline)
    with pytest.raises(ValueError, match="rank 0 must provide a tensor"):
        pipeline._broadcast_tensor(None, dtype=torch.long, device=_CPU)


# -----------------------------------------------------------------------------
# Payload routing of broadcast_tensor (mocked collectives, single process).
# -----------------------------------------------------------------------------

_DEVICE_GROUP = object()
_CPU_GROUP = object()


def _record_collectives(monkeypatch, *, src_values=None):
    """Record broadcast/synchronize calls; fill receive buffers like ``src``."""
    events = []
    received = list(src_values or [])

    def fake_broadcast(tensor, src, group):
        events.append(("broadcast", group, tensor.device, tensor.dtype, tensor.shape))
        if received:
            value = received.pop(0)
            if tensor.device.type != "meta":
                tensor.copy_(value)

    def fake_synchronize(*args):
        events.append(("synchronize",))

    monkeypatch.setattr(torch.distributed, "broadcast", fake_broadcast)
    monkeypatch.setattr(torch.accelerator, "synchronize", fake_synchronize)
    return events


def _header(shape, meta_len):
    return torch.tensor(
        [len(shape), *shape] + [0] * (meta_len - 1 - len(shape)), dtype=torch.int64
    )


@pytest.mark.parametrize("device", [_CPU, torch.device("meta")])
@pytest.mark.parametrize("is_src", [True, False])
def test_broadcast_tensor_payload_over_cpu_group(monkeypatch, is_src, device):
    expected = (torch.arange(12, dtype=torch.float32) * 0.25).reshape(3, 4)
    meta_len = minimax_h3.ENCODER_META_LEN
    events = _record_collectives(
        monkeypatch,
        src_values=None if is_src else [_header((3, 4), meta_len), expected],
    )
    out = minimax_h3.broadcast_tensor(
        expected if is_src else None,
        dtype=torch.bfloat16,
        device=device,
        src=3,
        is_src=is_src,
        device_group=_DEVICE_GROUP,
        cpu_group=_CPU_GROUP,
        meta_len=meta_len,
    )
    # Header and payload both go over the gloo group as host tensors; only
    # the final copy targets ``device``.
    assert events == [
        ("broadcast", _CPU_GROUP, _CPU, torch.int64, (meta_len,)),
        ("broadcast", _CPU_GROUP, _CPU, torch.bfloat16, (3, 4)),
    ]
    assert out.device == device
    assert out.dtype == torch.bfloat16
    assert out.shape == (3, 4)
    if device == _CPU:
        _assert_same(out, expected.to(torch.bfloat16))


def test_broadcast_tensor_source_already_on_host(monkeypatch):
    events = _record_collectives(monkeypatch)
    hidden = torch.linspace(-2.0, 2.0, 40).reshape(1, 5, 8).to(torch.bfloat16)
    out = minimax_h3.broadcast_tensor(
        hidden,
        dtype=torch.bfloat16,
        device=_CPU,
        src=0,
        is_src=True,
        device_group=_DEVICE_GROUP,
        cpu_group=_CPU_GROUP,
        meta_len=minimax_h3.DIT_META_LEN,
    )
    assert [event[1] for event in events] == [_CPU_GROUP, _CPU_GROUP]
    assert ("synchronize",) not in events
    _assert_same(out, hidden)


def test_broadcast_tensor_fallback_fenced_device_broadcast(monkeypatch):
    meta = torch.device("meta")
    events = _record_collectives(monkeypatch)
    # The header protocol is covered by the gloo tests; pin it here so the
    # payload can live on a non-CPU (meta) device.
    monkeypatch.setattr(minimax_h3, "broadcast_shape", lambda *a, **k: (2, 3))
    for is_src in (True, False):
        events.clear()
        out = minimax_h3.broadcast_tensor(
            torch.arange(6).reshape(2, 3) if is_src else None,
            dtype=torch.long,
            device=meta,
            src=0,
            is_src=is_src,
            device_group=_DEVICE_GROUP,
            cpu_group=None,
            meta_len=minimax_h3.DIT_META_LEN,
        )
        assert events == [
            ("synchronize",),
            ("broadcast", _DEVICE_GROUP, meta, torch.long, (2, 3)),
            ("synchronize",),
        ]
        assert out.device == meta
        assert out.dtype == torch.long
        assert out.shape == (2, 3)


def test_broadcast_int_list_fallback_fenced_device_broadcast(monkeypatch):
    meta = torch.device("meta")
    events = _record_collectives(monkeypatch)
    received = torch.tensor([3, -4, 5], dtype=torch.int32)
    # Meta tensors hold no data: stand in for the host read of the result.
    tensor_cpu = torch.Tensor.cpu
    monkeypatch.setattr(
        torch.Tensor,
        "cpu",
        lambda self, *a, **k: received if self.is_meta else tensor_cpu(self, *a, **k),
    )
    for values in ([3, -4, 5], None):
        events.clear()
        out = minimax_h3.broadcast_int_list(
            values,
            3,
            src=0,
            device_group=_DEVICE_GROUP,
            cpu_group=None,
            device=meta,
        )
        assert events == [
            ("synchronize",),
            ("broadcast", _DEVICE_GROUP, meta, torch.int32, (3,)),
            ("synchronize",),
        ]
        assert out == [3, -4, 5]


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


# -----------------------------------------------------------------------------
# (e) idempotency and no-op without vLLM-Omni
# -----------------------------------------------------------------------------


def test_minimax_h3_patches_idempotent(monkeypatch):
    pipeline = _fake_pipeline_module(None, None, 0, 1)
    monkeypatch.setitem(sys.modules, minimax_h3.PIPELINE_MODULE, pipeline)

    def patched_state():
        return (
            pipeline._broadcast_tensor,
            pipeline.dist,
            pipeline.MiniMaxH3Pipeline._encoder_group_broadcast_tensor,
            pipeline.MiniMaxH3Pipeline._distribute_encode_inputs,
        )

    assert minimax_h3.apply_minimax_h3_patches()
    state = patched_state()
    assert isinstance(pipeline.dist, minimax_h3.PipelineDist)
    assert pipeline._tpu_orig_dist is torch.distributed
    assert not minimax_h3.apply_minimax_h3_patches()
    assert patched_state() == state


def test_patches_noop_without_vllm_omni(monkeypatch):
    monkeypatch.setitem(sys.modules, minimax_h3.PIPELINE_MODULE, None)
    monkeypatch.setitem(
        sys.modules, "vllm_omni.diffusion.worker.diffusion_worker", None
    )
    assert minimax_h3.apply_minimax_h3_patches() is False
    omni_patches._patch_worker_proc_async_output_device()


def test_patches_skip_modules_that_are_not_loaded(monkeypatch):
    # apply_omni_model_specific_patches runs for every Omni model: MiniMax-H3
    # modules that are not already loaded must not be imported.
    h3_names = (minimax_h3.PIPELINE_MODULE,)
    for name in h3_names:
        monkeypatch.delitem(sys.modules, name, raising=False)
    import_module = importlib.import_module

    def guarded_import(name, package=None):
        assert name not in h3_names, f"unexpected import of {name}"
        return import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", guarded_import)
    assert minimax_h3.apply_minimax_h3_patches() is False
    assert all(name not in sys.modules for name in h3_names)
