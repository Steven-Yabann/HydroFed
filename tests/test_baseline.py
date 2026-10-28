import numpy as np

from hydrofed.baseline import train_baselines
from hydrofed.data.pipeline import save_partitions
from hydrofed.storage import ExperimentStore


def test_baseline_uses_held_out_plant_test_split(tmp_path):
    rng = np.random.default_rng(7)
    windows = rng.normal(size=(30, 4, 2)).astype(np.float32)

    # Train 0–17, calibrate 18–23, test 24–29.
    labels = np.array(
        [0] * 24 + [0, 1, 0, 1, 0, 1],
        dtype=np.int64,
    )

    partition_dir = tmp_path / "partitions"
    output_dir = tmp_path / "models"

    database_path = tmp_path / "experiments.db"

    save_partitions(
        {"plant_1": (windows, labels)},
        partition_dir,
    )

    results = train_baselines(
        partition_dir,
        output_dir,
        epochs=1,
        batch_size=4,
        db_path=database_path,
        model_config={
            "hidden_size": 4,
            "latent_size": 2,
            "num_layers": 1,
            "dropout": 0.0,
        },
    )

    metrics = results["plant_1"]

    assert (output_dir / "plant_1.pt").exists()
    assert 0.0 <= metrics["precision"] <= 1.0
    assert 0.0 <= metrics["recall"] <= 1.0
    assert 0.0 <= metrics["f1"] <= 1.0
    assert 0.0 <= metrics["auc_roc"] <= 1.0
    stored_runs = ExperimentStore(database_path).list_runs(
        experiment="baseline",
    )

    assert len(stored_runs) == 1
    assert stored_runs[0]["status"] == "completed"
    assert stored_runs[0]["result_count"] == 1

    stored_results = ExperimentStore(database_path).list_results(
        run_id=stored_runs[0]["run_id"],
    )

    assert len(stored_results) == 1
    assert stored_results[0]["client_id"] == "plant_1"