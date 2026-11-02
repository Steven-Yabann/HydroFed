"""Runtime helpers for selecting available compute hardware."""

import torch


def resolve_device(requested: str = "auto") -> str:
    """Return an available PyTorch device, preferring CUDA then Apple MPS."""
    if requested not in {"auto", "cpu", "cuda", "mps"}:
        raise ValueError("device must be 'auto', 'cpu', 'cuda', or 'mps'")

    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but no compatible GPU is available"
        )

    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError(
            "MPS was requested, but Apple Metal acceleration is unavailable"
        )

    return requested
