from flwr.common import parameters_to_ndarrays
from flwr.server.strategy import FedAvg
from hydrofed.storage import ExperimentStore

def parameter_bytes(parameters): return sum(array.nbytes for array in parameters_to_ndarrays(parameters))

class TrackingFedAvg(FedAvg):
    def __init__(self, db_path="artifacts/experiments.db", experiment="federated", **kwargs):
        super().__init__(**kwargs); self.store = ExperimentStore(db_path); self.experiment = experiment
    def aggregate_fit(self, server_round, results, failures):
        uplink = sum(parameter_bytes(result.parameters) for _, result in results)
        for _, result in results:
            self.store.record(self.experiment, "client_round", dict(result.metrics) | {"uplink_bytes": parameter_bytes(result.parameters)}, str(result.metrics.get("client_id", "unknown")), server_round)
        aggregated, metrics = super().aggregate_fit(server_round, results, failures)
        downlink = parameter_bytes(aggregated) * len(results) if aggregated is not None else 0
        self.store.record(self.experiment, "federation_round", {"uplink_bytes": uplink, "downlink_bytes": downlink, **(metrics or {})}, round_number=server_round)
        return aggregated, metrics
