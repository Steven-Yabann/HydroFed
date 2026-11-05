"""Condense the saved SKAB comparison results for the performance page.

``scripts/run_comparison.py`` writes detailed JSON results under
``artifacts/comparison``. This module picks out what the dashboard shows:
training regimes, heterogeneous and data-scarce scenarios, the model
families, and communication cost. Every detection figure is a mean and
standard deviation over three seeds, from the held-out final recordings
unless a caller asks for development recordings.
"""

from __future__ import annotations

import json
from pathlib import Path

METRICS = (
    "precision", "recall", "f1", "false_positive_rate", "macro_recording_auc_roc",
    "fault_events", "fault_events_detected", "median_detection_delay",
)
REGIME_LABELS = {
    "federated": "Federated (FedAvg)",
    "centralised": "Centralised",
    "isolated": "Isolated plants",
    "fedprox": "FedProx",
    "federated_finetune": "FedAvg + fine-tuning",
}
MODEL_LABELS = {
    "lstm_autoencoder": "LSTM autoencoder",
    "transformer_autoencoder_d32": "Transformer autoencoder (d=32)",
    "transformer_autoencoder_d64": "Transformer autoencoder (d=64)",
    "xgboost": "XGBoost",
    "federated_xgboost": "XGBoost (federated bagging)",
    "random_forest": "Random Forest",
    "isolation_forest": "Isolation Forest",
}
SCENARIO_LABELS = {
    "baseline": "Unchanged plants",
    "heterogeneous": "Heterogeneous plants",
    "scarce_100s": "Scarce data, 100 s",
    "scarce_50s": "Scarce data, 50 s",
    "scarce_25s": "Scarce data, 25 s",
}


def _load(directory: Path, name: str) -> dict | None:
    path = directory / f"{name}.json"
    return json.loads(path.read_text()) if path.exists() else None


def _metrics(summary: dict | None) -> dict | None:
    if not summary:
        return None
    return {metric: summary[metric] for metric in METRICS if metric in summary}


def _regimes(result: dict, role: str) -> dict:
    rows = []
    for label, scores in result["summary"].items():
        if label not in REGIME_LABELS:
            continue
        rows.append({
            "id": label,
            "label": REGIME_LABELS[label],
            "moves_raw_data": label == "centralised",
            "collaborates": label != "isolated",
            "full": _metrics(scores.get("hybrid", {}).get(role)),
            "model_only": _metrics(scores.get("residual", {}).get(role)),
        })
    rounds = []
    for label, scores in result["summary"].items():
        if label.startswith("federated_") and label.endswith("r"):
            rounds.append({
                "rounds": int(label.split("_")[1][:-1]),
                "full": _metrics(scores.get("hybrid", {}).get(role)),
                "model_only": _metrics(scores.get("residual", {}).get(role)),
            })
    rounds.append({
        "rounds": result["config"]["rounds"],
        "full": _metrics(result["summary"]["federated"]["hybrid"].get(role)),
        "model_only": _metrics(result["summary"]["federated"]["residual"].get(role)),
    })
    return {
        "rows": rows,
        "rounds": sorted(rounds, key=lambda item: item["rounds"]),
        "schedule": {key: result["config"][key] for key in ("rounds", "epochs", "seeds")},
    }


def _scenarios(*results: dict | None, role: str) -> dict | None:
    results = [result for result in results if result]
    if not results:
        return None
    groups = results[0]["groups"]
    scenarios = []
    for scenario in results[0]["summary"]:
        plant_groups = ["all"] if scenario in ("baseline", "heterogeneous") else ["scarce plants"]
        if scenario == "baseline":
            plant_groups.append("scarce plants")
        for group in plant_groups:
            regimes = []
            for result in results:
                for regime, scores in result["summary"].get(scenario, {}).items():
                    if regime == "no_model" or regime not in REGIME_LABELS:
                        continue
                    regimes.append({
                        "id": regime,
                        "label": REGIME_LABELS[regime],
                        "full": _auc(scores, "hybrid", role, group),
                        "model_only": _auc(scores, "residual", role, group),
                        "model_only_by_plant": {
                            plant: _auc(scores, "residual", role, plant)
                            for plant in sorted(groups) if plant.startswith("plant_")
                        } if group == "all" else None,
                    })
            no_model = None
            for result in results:
                if "no_model" in result["summary"].get(scenario, {}):
                    no_model = _auc(result["summary"][scenario]["no_model"], "levels", role, group)
            scenarios.append({
                "id": scenario,
                "label": SCENARIO_LABELS.get(scenario, scenario),
                "plants": "All plants" if group == "all" else ", ".join(groups.get(group, [group])),
                "regimes": _unique(regimes),
                "no_model_auc": no_model,
            })
    return {"scenarios": scenarios, "profiles": results[0].get("profiles", {})}


def _auc(scores: dict, score: str, role: str, group: str) -> float | None:
    """Within-recording AUC for all plants or one plant group."""
    if group == "all":
        summary = scores.get(score, {}).get(role)
    else:
        summary = scores.get("groups", {}).get(score, {}).get(group, {}).get(role)
    value = (summary or {}).get("macro_recording_auc_roc")
    return value["mean"] if isinstance(value, dict) else value


def _unique(rows: list[dict]) -> list[dict]:
    seen, unique = set(), []
    for row in rows:
        if row["id"] not in seen:
            seen.add(row["id"])
            unique.append(row)
    return unique


def _models(result: dict, role: str) -> dict:
    rows = []
    for name, candidate in result["candidates"].items():
        summary = candidate["summary"]
        score = "hybrid" if "hybrid" in summary else "model"
        rows.append({
            "id": name,
            "label": MODEL_LABELS.get(name, name),
            "category": candidate["category"],
            "training": candidate["training"],
            "needs_fault_labels": candidate["needs_fault_labels"],
            "selected": name in result["selected"].values(),
            "parameters": candidate.get("parameters"),
            "model_bytes": candidate.get("model_bytes"),
            "communication_bytes": candidate.get("communication_bytes"),
            "rounds": candidate.get("rounds"),
            "training_seconds": candidate.get("seconds"),
            "full": _metrics(summary.get(score, {}).get(role)),
            "model_only": _metrics(summary.get("residual", {}).get(role)) if "residual" in summary else None,
            "unseen_fault_type": _metrics(summary.get(score, {}).get("final_unseen_fault_type")),
        })
    return {"rows": rows, "selected": result["selected"]}


def _communication(result: dict, models: dict | None) -> dict:
    communication = result["communication"]
    federated = [
        {"rounds": int(name.split("_")[0]), **traffic}
        for name, traffic in communication["federated"].items()
    ]
    per_model = []
    if models:
        for name, candidate in models["candidates"].items():
            if candidate.get("communication_bytes"):
                per_model.append({
                    "id": name,
                    "label": MODEL_LABELS.get(name, name),
                    "training": candidate["training"],
                    "rounds": candidate.get("rounds"),
                    "model_bytes": candidate.get("model_bytes"),
                    "communication_bytes": candidate["communication_bytes"],
                })
    return {
        "model_bytes": communication["model_bytes"],
        "clients": communication["clients"],
        "raw": communication["raw_skab"],
        "streaming_bytes_per_plant_day_1hz": communication["streaming_bytes_per_plant_day_1hz"],
        "federated": sorted(federated, key=lambda item: item["rounds"]),
        "per_model": per_model,
    }


def comparison_digest(directory: str | Path, role: str = "final") -> dict:
    if role not in ("dev", "final"):
        raise ValueError("role must be 'dev' or 'final'")
    directory = Path(directory)
    regimes = _load(directory, "regimes")
    models = _load(directory, "models")
    scenarios = _load(directory, "scenarios")
    personalised = _load(directory, "personalised")
    return {
        "role": role,
        "available": {
            "regimes": regimes is not None,
            "models": models is not None,
            "scenarios": scenarios is not None,
            "personalised": personalised is not None,
        },
        "regimes": _regimes(regimes, role) if regimes else None,
        "scenarios": _scenarios(scenarios, personalised, role=role),
        "models": _models(models, role) if models else None,
        "communication": _communication(regimes, models) if regimes else None,
    }
