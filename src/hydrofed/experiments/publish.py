"""Save comparison summaries as experiment runs the API and dashboard can read.

Each comparison becomes one run with flat result rows:

regime_comparison
    ``detection``: one row per regime, score and role (mean and sd per metric)
    ``communication``: federated traffic for each number of rounds
    ``raw_data``: what centralising the SKAB recordings would transfer

model_comparison
    ``candidate``: how each model trains, its size, cost and selection
    ``detection``: one row per model, score and role
"""

from __future__ import annotations

import re

from hydrofed.storage import ExperimentStore

REGIME_EXPERIMENT = "regime_comparison"
MODEL_EXPERIMENT = "model_comparison"
CANDIDATE_FIELDS = (
    "category", "training", "needs_fault_labels", "parameters", "model_bytes",
    "communication_bytes", "uplink_bytes", "downlink_bytes", "rounds", "clients", "trees",
)


def regime_rows(result: dict) -> list[dict]:
    """Return result rows for a ``run_comparison.py regimes`` result."""
    default_rounds = result["config"]["rounds"]
    rows = []
    for label, scores in result["summary"].items():
        match = re.fullmatch(r"(\w+?)_(\d+)r", label)
        regime, rounds = (match.group(1), int(match.group(2))) if match else (label, default_rounds)
        for score, roles in scores.items():
            for role, metrics in roles.items():
                rows.append({
                    "kind": "detection", "client_id": label,
                    "metrics": {"regime": regime, "rounds": rounds, "score": score, "role": role, **metrics},
                })

    communication = result["communication"]
    for name, traffic in communication["federated"].items():
        rows.append({
            "kind": "communication", "client_id": name,
            "metrics": {"rounds": int(name.split("_")[0]), **traffic},
        })
    rows.append({
        "kind": "raw_data", "client_id": "skab",
        "metrics": {
            **communication["raw_skab"],
            "model_bytes": communication["model_bytes"],
            "clients": communication["clients"],
            "streaming_bytes_per_plant_day_1hz": communication["streaming_bytes_per_plant_day_1hz"],
        },
    })
    return rows


def model_rows(result: dict) -> list[dict]:
    """Return result rows for a ``run_comparison.py models`` result."""
    selected = set(result["selected"].values())
    rows = []
    for name, candidate in result["candidates"].items():
        dev = candidate["summary"][candidate["selection_score"]]["dev"]
        rows.append({
            "kind": "candidate", "client_id": name,
            "metrics": {
                **{field: candidate[field] for field in CANDIDATE_FIELDS if field in candidate},
                "selected": name in selected,
                "selection_score": candidate["selection_score"],
                "dev_auc": dev["macro_recording_auc_roc"]["mean"],
            },
        })
        for score, roles in candidate["summary"].items():
            for role, metrics in roles.items():
                rows.append({
                    "kind": "detection", "client_id": name,
                    "metrics": {
                        "category": candidate["category"], "training": candidate.get("training"),
                        "score": score, "role": role, **metrics,
                    },
                })
    return rows


def publish(store: ExperimentStore, experiment: str, config: dict, rows: list[dict]) -> str:
    """Save rows as one completed run and return its run ID."""
    run_id = store.create_run(experiment, config)
    try:
        for row in rows:
            store.record(experiment, row["kind"], row["metrics"], client_id=row["client_id"], run_id=run_id)
    except Exception as error:
        store.complete_run(run_id, "failed", str(error))
        raise
    store.complete_run(run_id)
    return run_id
