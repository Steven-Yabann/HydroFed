from prometheus_client import generate_latest

from hydrofed.monitoring import COMMUNICATION_BYTES, LOCAL_LOSS, ROUND


def test_hydrofed_metrics_are_exported_in_prometheus_format():
    ROUND.set(2)
    LOCAL_LOSS.labels("plant_1").set(0.25)
    COMMUNICATION_BYTES.labels("uplink").set(1024)

    exported = generate_latest().decode("utf-8")

    assert "hydrofed_round 2.0" in exported
    assert 'hydrofed_local_loss{client_id="plant_1"} 0.25' in exported
    assert 'hydrofed_communication_bytes{direction="uplink"} 1024.0' in exported
