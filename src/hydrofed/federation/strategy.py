from flwr.common import parameters_to_ndarrays
from flwr.server.strategy import FedAvg
from hydrofed.monitoring import (ANOMALY_SCORE, COMMUNICATION_BYTES, F1,
                                 GLOBAL_LOSS, LOCAL_LOSS, ROUND)
from hydrofed.storage import ExperimentStore

def parameter_bytes(parameters): return sum(array.nbytes for array in parameters_to_ndarrays(parameters))

class TrackingFedAvg(FedAvg):
    def __init__(
        self,
        db_path="artifacts/experiments.db",
        experiment="federated",
        run_id: str | None = None,
        **kwargs,
    ):
        initial_parameters = kwargs.get("initial_parameters")

        super().__init__(**kwargs)

        self.store = ExperimentStore(db_path)
        self.experiment = experiment
        self.run_id = run_id
        self.final_parameters = initial_parameters

    def aggregate_fit(self, server_round, results, failures):
        uplink = sum(parameter_bytes(result.parameters) for _, result in results)
        for _, result in results:
            self.store.record(
                self.experiment,
                "client_round",
                dict(result.metrics)
                | {"uplink_bytes": parameter_bytes(result.parameters)},
                client_id=str(result.metrics.get("client_id", "unknown")),
                round_number=server_round,
                run_id=self.run_id,
            )
            client_id = str(result.metrics.get("client_id", "unknown"))
            if "local_loss" in result.metrics:
                LOCAL_LOSS.labels(client_id).set(float(result.metrics["local_loss"]))
        aggregated, metrics = super().aggregate_fit(
            server_round,
            results,
            failures
        )
        if aggregated is not None:
            self.final_parameters = aggregated

        downlink = parameter_bytes(aggregated) * len(results) if aggregated is not None else 0
        ROUND.set(server_round)
        COMMUNICATION_BYTES.labels("uplink").set(uplink)
        COMMUNICATION_BYTES.labels("downlink").set(downlink)
        self.store.record(
            self.experiment,
            "federation_round",
            {
                "uplink_bytes": uplink,
                "downlink_bytes": downlink,
                **(metrics or {}),
            },
            round_number=server_round,
            run_id=self.run_id,
        )
        return aggregated, metrics

    def get_final_parameters_numpy(self):
        """Return the latest global model weights as NumPy arrays."""
        if self.final_parameters is None:
            raise RuntimeError("No global model parameters are available")

        return parameters_to_ndarrays(self.final_parameters)

    def aggregate_evaluate(self, server_round, results, failures):
        """Record evaluation values from Flower workers in server-visible gauges."""
        loss, metrics = super().aggregate_evaluate(server_round, results, failures)
        if loss is not None:
            GLOBAL_LOSS.set(loss)
        for _, result in results:
            client_id = str(result.metrics.get("client_id", "unknown"))
            if "local_loss" in result.metrics:
                ANOMALY_SCORE.labels(client_id).set(float(result.metrics["local_loss"]))
            if "f1" in result.metrics:
                F1.set(float(result.metrics["f1"]))
        return loss, metrics
