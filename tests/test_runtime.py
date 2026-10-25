import pytest

from hydrofed.runtime import resolve_device


def test_auto_uses_cuda_when_available(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)

    assert resolve_device() == "cuda"


def test_auto_uses_cpu_without_cuda(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)

    assert resolve_device() == "cpu"


def test_cpu_can_be_requested_explicitly():
    assert resolve_device("cpu") == "cpu"


def test_cuda_request_fails_without_gpu(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)

    with pytest.raises(RuntimeError, match="CUDA was requested"):
        resolve_device("cuda")


def test_unknown_device_is_rejected():
    with pytest.raises(ValueError, match="device must be"):
        resolve_device("tpu")