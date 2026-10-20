import pytest

from hydrofed.config import load_experiment_config


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