"""Start federated training runs from the dashboard.

An administrator's saved federation settings become arguments to
``scripts/run_experiment.py``, which runs as a separate process and records
its own experiment run, rounds and evaluation in the database. Only one
training process runs at a time. The API image used for deployment does not
ship the training scripts or data, so launching is refused there.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = PROJECT_ROOT / "scripts" / "run_experiment.py"
DEFAULT_PARTITIONS = PROJECT_ROOT / "data" / "partitions_skab"


class JobError(Exception):
    """A training run that cannot be started."""


def experiment_arguments(config: dict, clients: list[str], database: Path, partitions: Path, log_dir: Path) -> list[str]:
    """Translate dashboard federation settings into run_experiment.py arguments."""
    stamp = time.strftime("%Y%m%d-%H%M%S")
    return [
        sys.executable, str(SCRIPT),
        "--partitions", str(partitions),
        "--clients", ",".join(clients),
        "--fraction-fit", str(config["fraction_fit"]),
        "--rounds", str(config["rounds"]),
        "--epochs", str(config["local_epochs"]),
        "--batch-size", str(config["batch_size"]),
        "--learning-rate", str(config["learning_rate"]),
        "--hidden-size", str(config["hidden_size"]),
        "--latent-size", str(config["latent_size"]),
        "--num-layers", str(config["num_layers"]),
        "--dropout", str(config["dropout"]),
        "--seed", str(config["seed"]),
        "--device", config["device"],
        "--db", str(database),
        "--output", str(log_dir / f"dashboard_{stamp}.pt"),
        "--metrics-port", os.environ.get("HYDROFED_TRAINING_METRICS_PORT", "9101"),
        "--evaluate",
        "--evaluation-protocol", "skab",
        "--skab-scaling", "plant",
        "--skab-score", "hybrid",
        "--skab-training-policy", "normal-only",
        "--skab-exclude-anomaly-free",
    ]


class TrainingLauncher:
    def __init__(self, database: Path, partitions: Path | None = None, log_dir: Path | None = None):
        self.database = Path(database)
        self.partitions = Path(partitions or os.environ.get("HYDROFED_PARTITIONS", DEFAULT_PARTITIONS))
        self.log_dir = Path(log_dir or self.database.parent / "training_logs")
        self._lock = threading.Lock()
        self._process: subprocess.Popen | None = None
        self._job: dict | None = None

    def available(self) -> tuple[bool, str]:
        if os.environ.get("HYDROFED_ALLOW_TRAINING", "1") != "1":
            return False, "Training from the dashboard is switched off on this server."
        if not SCRIPT.exists():
            return False, "This server does not include the training scripts."
        if not any(self.partitions.glob("plant_*.npz")):
            return False, f"No plant partitions found under {self.partitions}."
        return True, ""

    def status(self) -> dict:
        available, reason = self.available()
        job = dict(self._job) if self._job else None
        if job and self._process is not None:
            code = self._process.poll()
            job["state"] = "running" if code is None else ("completed" if code == 0 else "failed")
            job["exit_code"] = code
            job["log_tail"] = self.log_tail()
        return {"available": available, "reason": reason, "job": job}

    def log_tail(self, lines: int = 40) -> list[str]:
        if not self._job:
            return []
        path = Path(self._job["log_path"])
        if not path.exists():
            return []
        return path.read_text(errors="replace").splitlines()[-lines:]

    def start(self, config: dict, clients: list[str], user: dict) -> dict:
        available, reason = self.available()
        if not available:
            raise JobError(reason)
        if len(clients) < 2:
            raise JobError("Federated training needs at least two participating plants.")
        with self._lock:
            if self._process is not None and self._process.poll() is None:
                raise JobError("A training run is already in progress.")
            self.log_dir.mkdir(parents=True, exist_ok=True)
            arguments = experiment_arguments(config, clients, self.database, self.partitions, self.log_dir)
            log_path = self.log_dir / f"run_{time.strftime('%Y%m%d-%H%M%S')}.log"
            log = log_path.open("w")
            self._process = subprocess.Popen(
                arguments, cwd=PROJECT_ROOT, stdout=log, stderr=subprocess.STDOUT,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )
            log.close()
            self._job = {
                "pid": self._process.pid,
                "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "started_by": user["name"],
                "clients": clients,
                "config": config,
                "log_path": str(log_path),
            }
        return self.status()

    def stop(self) -> dict:
        with self._lock:
            if self._process is None or self._process.poll() is not None:
                raise JobError("No training run is in progress.")
            self._process.terminate()
            try:
                self._process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._process.kill()
        return self.status()
