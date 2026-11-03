"""Exchange PyTorch model weights as NumPy arrays, as Flower does."""

from __future__ import annotations

from collections import OrderedDict

import numpy as np
import torch


class NumpyParametersMixin:
    """Add Flower-friendly weight access to an ``nn.Module``."""

    def get_parameters_numpy(self) -> list[np.ndarray]:
        """Return model weights as NumPy arrays.

        Flower exchanges model parameters as NumPy arrays, so this method
        converts the PyTorch state dictionary into Flower-friendly values.
        """

        return [value.detach().cpu().numpy() for value in self.state_dict().values()]

    def set_parameters_numpy(self, parameters: list[np.ndarray]) -> None:
        """Load model weights from NumPy arrays."""

        current_state = self.state_dict()

        if len(parameters) != len(current_state):
            raise ValueError(
                f"Expected {len(current_state)} parameter arrays, "
                f"but received {len(parameters)}"
            )

        new_state = OrderedDict()

        for key, value in zip(current_state.keys(), parameters):
            current_tensor = current_state[key]
            new_state[key] = torch.as_tensor(
                value,
                dtype=current_tensor.dtype,
                device=current_tensor.device,
            )

        self.load_state_dict(new_state, strict=True)

    def count_trainable_parameters(self) -> int:
        """Return the number of trainable parameters in the model."""

        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
