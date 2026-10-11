"""Tests for Prometheus monitoring endpoints."""

from fastapi.testclient import TestClient


def test_metrics_endpoint_returns_prometheus_format(client: TestClient):
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]
    body = response.text
    assert "demandpulse_http_requests_total" in body
    assert "demandpulse_forecast_requests_total" in body


def test_metrics_counter_increments_after_request(client: TestClient):
    client.get("/health")
    response = client.get("/metrics")
    assert response.status_code == 200
    body = response.text
    # The /health endpoint should have been counted
    assert 'path="/health"' in body


def test_prediction_log_write(tmp_path, monkeypatch):
    """log_prediction should not raise and should append to a JSONL file."""
    from app.monitoring import predictions

    monkeypatch.setattr(predictions, "_LOG_PATH", tmp_path / "predictions.jsonl")
    predictions.log_prediction({"product": "P0001", "horizon": 30})

    log_file = tmp_path / "predictions.jsonl"
    assert log_file.exists()
    content = log_file.read_text(encoding="utf-8").strip()
    assert "P0001" in content
