# Notebooks

## colab_ablation.ipynb

The Google Colab notebook used for GPU runs, saved with its outputs. The
partition ablation ran on 1 November 2026. It used an earlier version of the pipeline, so none of the results
reported in `docs/comparison.md` come from it.

What it ran:

- **Setup (cells 0–9).** A Python 3.11 environment built with `uv`, PyTorch
  2.2.2 for CUDA 12.1, and the repository at branch
  `feature/colab-experiments`. Cell 0's output was cleared: it came from a
  first attempt on a CPU runtime and only showed the GPU check failing. Cell 8,
  which printed the versions in the 3.11 environment, has no saved output.
- **Early runs (cells 10–14).** Isolated baselines for 100 epochs and Flower
  FedAvg for 100 rounds of 1 epoch, seeds 42 and 43, on `data/partitions` with
  the reconstruction score and a 95th-percentile threshold. Outputs went to
  `MyDrive/HydroFed/colab_runs/`.
- **Partition ablation (cells 16–26).** Branch
  `feature/balanced-partition-control`: recording-aware temporal partitions
  against an IID control, both with at least 100 anomalous windows per plant.
  Federated runs used 5 rounds of 5 epochs and isolated baselines 25 epochs,
  with latent size 16 and a 95th-percentile threshold. Outputs went to
  `MyDrive/HydroFed/ablation/`.

Seed naming: cells 24 and 25 ran seed 44 but saved to `temporal_seed43.db` and
`iid_seed43.db`. Each of those files therefore holds two federated runs, one
with seed 43 and one with seed 44, as cell 26's output shows. Read the seed
from each run's `config_json`, not from the file name.

The Drive outputs are attached to the GitHub release.
