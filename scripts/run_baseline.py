#!/usr/bin/env python3
import argparse, json
from hydrofed.baseline import train_baselines
p=argparse.ArgumentParser(); p.add_argument("--partitions",default="data/partitions"); p.add_argument("--output",default="artifacts/baselines"); p.add_argument("--epochs",type=int,default=1); p.add_argument("--db",default="artifacts/experiments.db")
p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
p.add_argument("--seed", type=int, default=42)
a=p.parse_args()

print(json.dumps(
    train_baselines(
        a.partitions,
        a.output,
        a.epochs,
        db_path=a.db,
        device=a.device,
        seed=a.seed,
    ),
    indent=2,
))
