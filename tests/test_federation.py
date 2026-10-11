import numpy as np
from hydrofed.federation.client import HydroFedClient
from hydrofed.data.pipeline import save_partitions

def test_client_fit_and_evaluate_contract(tmp_path):
    rng=np.random.default_rng(2); windows=rng.normal(size=(12,6,2)).astype(np.float32)
    labels=np.array([0]*9+[1]*3); save_partitions({"plant_1":(windows,labels)},tmp_path)
    client=HydroFedClient("plant_1",tmp_path/"plant_1.npz",{"hidden_size":4,"latent_size":2,"num_layers":1,"dropout":0},batch_size=4)
    parameters=client.get_parameters({}); updated,count,metrics=client.fit(parameters,{"local_epochs":1})
    loss,evaluated,evaluation_metrics=client.evaluate(updated,{})
    assert count==9 and len(updated)==len(parameters) and metrics["client_id"]=="plant_1"
    assert evaluated==12 and loss>=0 and evaluation_metrics["local_loss"]==loss
