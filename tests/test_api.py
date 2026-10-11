"""Integration tests for HTTP endpoints."""

import io

from fastapi.testclient import TestClient


def test_root_returns_html(client: TestClient):
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "DemandPulse" in response.text


def test_health_endpoint(client: TestClient):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "version" in body


def test_health_live(client: TestClient):
    response = client.get("/health/live")
    assert response.status_code == 200
    assert response.json() == {"status": "alive"}


def test_health_ready(client: TestClient):
    response = client.get("/health/ready")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"
    assert body["checks"]["data_dir"] is True


def test_request_id_header_present(client: TestClient):
    response = client.get("/health")
    assert "x-request-id" in response.headers
    assert "x-process-time-ms" in response.headers


def test_upload_rejects_missing_file(client: TestClient):
    response = client.post("/api/upload")
    assert response.status_code in {400, 422}


def test_upload_rejects_invalid_extension(client: TestClient):
    files = {"file": ("bad.txt", io.BytesIO(b"hello"), "text/plain")}
    response = client.post("/api/upload", files=files)
    assert response.status_code == 400
    assert "Invalid file type" in response.json()["detail"]


def test_forecast_requires_session(client: TestClient):
    payload = {"session_id": "does-not-exist", "horizon": 30}
    response = client.post("/api/forecast", json=payload)
    assert response.status_code == 404


def test_forecast_rejects_invalid_horizon(client: TestClient):
    payload = {"session_id": "any", "horizon": 5}
    response = client.post("/api/forecast", json=payload)
    assert response.status_code == 422
