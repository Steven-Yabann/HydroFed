import flwr as fl
from hydrofed.data.pipeline import load_client_partition
from hydrofed.models.lstm_autoencoder import build_model
from hydrofed.training.local import train_epochs, reconstruction_errors

class HydroFedClient(fl.client.NumPyClient):
    def __init__(self, client_id, partition_path, model_config=None, epochs=1, batch_size=64):
        self.client_id = str(client_id); self.windows, self.labels = load_client_partition(partition_path)
        self.model = build_model(self.windows.shape[2], model_config); self.epochs, self.batch_size = epochs, batch_size
    def get_parameters(self, config): return self.model.get_parameters_numpy()
    def fit(self, parameters, config):
        self.model.set_parameters_numpy(parameters); train = self.windows[self.labels == 0]
        loss = train_epochs(self.model, train, int(config.get("local_epochs", self.epochs)), self.batch_size)
        return self.model.get_parameters_numpy(), len(train), {"client_id": self.client_id, "local_loss": loss}
    def evaluate(self, parameters, config):
        self.model.set_parameters_numpy(parameters); loss = float(reconstruction_errors(self.model, self.windows).mean())
        return loss, len(self.windows), {"client_id": self.client_id, "local_loss": loss}
