import sys
from types import SimpleNamespace

import pytest

from moxhimt_runner.server_cuda import validate_cuda_runtime
from moxhimt_runner.server_rocm import validate_rocm_runtime


class AvailableAccelerator:
    @staticmethod
    def is_available() -> bool:
        return True


def fake_torch(*, cuda_version=None, hip_version=None):
    return SimpleNamespace(
        cuda=AvailableAccelerator(),
        version=SimpleNamespace(cuda=cuda_version, hip=hip_version),
    )


def test_cuda_runtime_accepts_cuda_and_rejects_rocm(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", fake_torch(cuda_version="12.8"))
    validate_cuda_runtime()

    monkeypatch.setitem(sys.modules, "torch", fake_torch(hip_version="7.1"))
    with pytest.raises(RuntimeError, match="ROCm"):
        validate_cuda_runtime()


def test_rocm_runtime_accepts_hip_and_rejects_cuda(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", fake_torch(hip_version="7.1"))
    validate_rocm_runtime()

    monkeypatch.setitem(sys.modules, "torch", fake_torch(cuda_version="12.8"))
    with pytest.raises(RuntimeError, match="ROCm/HIP"):
        validate_rocm_runtime()
