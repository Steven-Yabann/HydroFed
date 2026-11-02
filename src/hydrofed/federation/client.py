import flwr as fl
from hydrofed.data.pipeline import load_client_partition
from hydrofed.models.lstm_autoencoder import build_model
from hydrofed.training.local import train_epochs, reconstruction_errors
from hydrofed.data.splits import split_plant_windows
from hydrofed.data.scaling import WindowMinMaxScaler, scale_plant_split
from hydrofed.runtime import resolve_device
from hydrofed.reproducibility import set_random_seed

class HydroFedClient(fl.client.NumPyClient):
    def __init__(
    self,
    client_id,
    partition_path,
    model_config=None,
    epochs=1,
    batch_size=64,
    device: str = "auto",
    seed: int = 42,
    scaler: WindowMinMaxScaler | None = None,
    ):
        self.client_id = str(client_id); self.windows, self.labels = load_client_partition(partition_path)
        self.model = build_model(self.windows.shape[2], model_config); self.epochs, self.batch_size = epochs, batch_size
        self.plant_split = split_plant_windows(
            self.windows,
            self.labels,
        )
        if scaler is not None:
            self.plant_split = scale_plant_split(self.plant_split, scaler)
        self.device = resolve_device(device)
        self.seed = seed


    def get_parameters(self, config): return self.model.get_parameters_numpy()

    def fit(self, parameters, config):
        set_random_seed(self.seed)
        self.model.set_parameters_numpy(parameters)
        self.model.set_parameters_numpy(parameters)

        loss = train_epochs(
            self.model,
            self.plant_split.train_windows,
            int(config.get("local_epochs", self.epochs)),
            self.batch_size,
            device=self.device,
        )

        return (
            self.model.get_parameters_numpy(),
            len(self.plant_split.train_windows),
            {"client_id": self.client_id, "local_loss": loss},
        )

    def evaluate(self, parameters, config):
        self.model.set_parameters_numpy(parameters)

        loss = float(
            reconstruction_errors(
                self.model,
                self.plant_split.test_windows,
                device=self.device,
            ).mean()
        )

        return (
            loss,
            len(self.plant_split.test_windows),
            {"client_id": self.client_id, "local_loss": loss},
        )
