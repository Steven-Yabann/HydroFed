"""Utilities for repeatable experiments."""

from __future__ import annotations

import random

import numpy as np
import torch


def set_random_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch random number generators."""
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError("seed must be an integer")

    if seed < 0:
        raise ValueError("seed must be non-negative")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)