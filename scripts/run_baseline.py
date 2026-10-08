#!/usr/bin/env python3
import argparse, json
from hydrofed.baseline import train_baselines
p=argparse.ArgumentParser(); p.add_argument("--partitions",default="data/partitions"); p.add_argument("--output",default="artifacts/baselines"); p.add_argument("--epochs",type=int,default=1); p.add_argument("--db",default="artifacts/experiments.db")
a=p.parse_args(); print(json.dumps(train_baselines(a.partitions,a.output,a.epochs,db_path=a.db),indent=2))
