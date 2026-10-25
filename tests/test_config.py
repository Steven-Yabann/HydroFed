import random

import numpy as np
import pytest
import torch

from hydrofed.config import load_experiment_config
from hydrofed.reproducibility import set_random_seed


def test_load_experiment_config_reads_nested_yaml_values(tmp_path):
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(
        "seed: 7\n"
        "training:\n"
        "  local_epochs: 3\n"
    )

    config = load_experiment_config(config_path)

    assert config["seed"] == 7
    assert config["training"]["local_epochs"] == 3


def test_load_experiment_config_rejects_non_mapping_yaml(tmp_path):
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text("- not\n- a mapping\n")

    with pytest.raises(ValueError, match="YAML mapping"):
        load_experiment_config(config_path)

def test_set_random_seed_repeats_python_numpy_and_torch_values():
    set_random_seed(42)

    first_python = random.random()
    first_numpy = np.random.random()
    first_torch = torch.rand(3)

    set_random_seed(42)

    assert random.random() == first_python
    assert np.random.random() == first_numpy
    assert torch.equal(torch.rand(3), first_torch)