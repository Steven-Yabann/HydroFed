#!/usr/bin/env python3
import argparse, json, torch
import numpy as np
from hydrofed.data.pipeline import load_client_partition
from hydrofed.data.scaling import WindowMinMaxScaler
from hydrofed.models.lstm_autoencoder import build_model
from hydrofed.evaluation import evaluate_model
p=argparse.ArgumentParser(); p.add_argument("weights"); p.add_argument("partition"); a=p.parse_args()
x,y=load_client_partition(a.partition); checkpoint=torch.load(a.weights,map_location="cpu",weights_only=True)
preprocessing = checkpoint.get("preprocessing", {})
if preprocessing.get("scaler") == "shared_normal_train_minmax":
    scaler = WindowMinMaxScaler(
        minimums=np.asarray(preprocessing["minimums"], dtype=np.float32),
        spans=np.asarray(preprocessing["spans"], dtype=np.float32),
    )
    x = scaler.transform(x)
model=build_model(checkpoint.get("n_features",x.shape[2])); model.load_state_dict(checkpoint["state_dict"]); result=evaluate_model(model,x,y,checkpoint["threshold"]); print(json.dumps(result,indent=2))
