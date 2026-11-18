#!/usr/bin/env python3
"""Tabulate the FL strategy screen next to the scenario regimes.

Usage: report_fl_strategy_screen.py SCREEN_JSON [SCENARIOS_JSON]
SCENARIOS_JSON defaults to artifacts/comparison/scenarios.json.
"""
import json
import sys

new = json.load(open(sys.argv[1]))["summary"]
reference = sys.argv[2] if len(sys.argv) > 2 else "artifacts/comparison/scenarios.json"
ref = json.load(open(reference))["summary"]
REF_GROUPS = {"all": None, "scarce plants": "scarce plants", "other plants": "other plants"}


def ref_metrics(scenario, regime, score, group, role):
    entry = ref[scenario][regime]
    return entry[score][role] if group == "all" else entry["groups"][score][group][role]


def fmt(m, name):
    value = m[name]["mean"]
    return f"{value:.3f}" if value is not None else " n/a "


methods = ["fedavg", "fedavg_finetune", "median", "trimmed", "fedprox", "clustered"]
for scenario in ("baseline", "heterogeneous", "scarce_25s"):
    group = "scarce plants" if scenario == "scarce_25s" else "all"
    print(f"\n### {scenario} ({group})")
    print("| Method | Model-only AUC dev | Model-only AUC final | Full AUC final | Full recall final | Full false alarms final |")
    print("|---|---|---|---|---|---|")
    rows = [(f"ref {r}", lambda s, g, ro, r=r: ref_metrics(scenario, r, s, g, ro)) for r in ("centralised", "isolated", "federated")]
    rows += [(m, lambda s, g, ro, m=m: new[f"{scenario}/{m}"][s][g][ro]) for m in methods]
    for name, get in rows:
        print(f"| {name} | {fmt(get('residual', group, 'dev'), 'macro_recording_auc_roc')} | "
              f"{fmt(get('residual', group, 'final'), 'macro_recording_auc_roc')} | "
              f"{fmt(get('hybrid', group, 'final'), 'macro_recording_auc_roc')} | "
              f"{fmt(get('hybrid', group, 'final'), 'recall')} | {fmt(get('hybrid', group, 'final'), 'false_positive_rate')} |")
    if scenario == "heterogeneous":
        print("\nModel-only AUC per plant, final")
        plants = ["plant_1", "plant_2", "plant_3", "plant_4", "plant_5"]
        print("| Method | " + " | ".join(plants) + " |")
        print("|---" * 6 + "|")
        for name in ("isolated",):
            print(f"| ref {name} | " + " | ".join(fmt(ref[scenario][name]["groups"]["residual"][p]["final"], "macro_recording_auc_roc") for p in plants) + " |")
        for m in methods:
            print(f"| {m} | " + " | ".join(fmt(new[f"{scenario}/{m}"]["residual"][p]["final"], "macro_recording_auc_roc") for p in plants) + " |")
    clusters = new.get(f"{scenario}/clustered", {}).get("clusters")
    print("clusters:", clusters)
