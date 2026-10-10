"""Prometheus metrics used by experiment processes."""
from prometheus_client import Gauge, start_http_server
ROUND = Gauge("hydrofed_round", "Current federation round")
LOCAL_LOSS = Gauge("hydrofed_local_loss", "Latest local loss", ["client_id"])
GLOBAL_LOSS = Gauge("hydrofed_global_loss", "Latest global loss")
ANOMALY_SCORE = Gauge("hydrofed_anomaly_score", "Mean anomaly score", ["client_id"])
COMMUNICATION_BYTES = Gauge("hydrofed_communication_bytes", "Parameter bytes", ["direction"])
F1 = Gauge("hydrofed_f1", "Latest F1 score")
def serve(port=8000): start_http_server(port)
