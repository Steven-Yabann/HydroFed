#!/usr/bin/env python3
import argparse
from pathlib import Path
import flwr as fl
from flwr.common import ndarrays_to_parameters
from hydrofed.data.pipeline import load_client_partition
from hydrofed.federation.client import HydroFedClient
from hydrofed.federation.strategy import TrackingFedAvg
from hydrofed.models.lstm_autoencoder import build_model
p=argparse.ArgumentParser(); p.add_argument("--partitions",default="data/partitions"); p.add_argument("--rounds",type=int,default=2); p.add_argument("--epochs",type=int,default=1); p.add_argument("--db",default="artifacts/experiments.db")
a=p.parse_args(); files=sorted(Path(a.partitions).glob("plant_*.npz"))
if not files: raise SystemExit("No plant partitions found; run make partition-synthetic")
n_features=load_client_partition(files[0])[0].shape[2]
def client_fn(cid): return HydroFedClient(cid, files[int(cid)], epochs=a.epochs).to_client()
strategy=TrackingFedAvg(db_path=a.db, fraction_fit=1.0, fraction_evaluate=1.0, min_available_clients=len(files), initial_parameters=ndarrays_to_parameters(build_model(n_features).get_parameters_numpy()))
fl.simulation.start_simulation(client_fn=client_fn,num_clients=len(files),config=fl.server.ServerConfig(num_rounds=a.rounds),strategy=strategy,client_resources={"num_cpus":1})
