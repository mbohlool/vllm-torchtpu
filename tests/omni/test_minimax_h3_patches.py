# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the MiniMax-H3 / vLLM-Omni TPU patches.

These tests never import vLLM-Omni or MiniMax-H3 code.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm_torchtpu.omni import patches as omni_patches

pytestmark = pytest.mark.omni

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
