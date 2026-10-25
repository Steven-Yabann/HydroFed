"""Runtime helpers for selecting available compute hardware."""

import torch


def resolve_device(requested: str = "auto") -> str:
    """Return the requested PyTorch device, or choose CUDA when available."""
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be 'auto', 'cpu', or 'cuda'")

    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"

    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but no compatible GPU is available"
        )

    return requested