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

def test_reported_settings_match_the_comparison_defaults():
    import argparse
    import importlib.util
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("run_comparison", root / "scripts" / "run_comparison.py")
    run_comparison = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run_comparison)
    parser = argparse.ArgumentParser()
    run_comparison.add_common_arguments(parser)
    defaults = vars(parser.parse_args([]))

    config = load_experiment_config(root / "configs" / "skab_experiment.yaml")

    assert defaults["seeds"] == config["seeds"]
    assert defaults["partitions"] == config["data"]["partitions"]
    assert defaults["scaling"] == config["data"]["scaling"]
    for name in ("hidden_size", "latent_size", "num_layers", "dropout"):
        assert defaults[name] == config["model"][name]
    assert defaults["epochs"] == config["training"]["local_epochs"]
    assert defaults["batch_size"] == config["training"]["batch_size"]
    assert defaults["learning_rate"] == config["training"]["learning_rate"]
    assert defaults["rounds"] == config["federation"]["rounds"]
    for name in ("smoothing", "calibration_tail", "persistence"):
        assert defaults[name] == config["alerts"][name]
