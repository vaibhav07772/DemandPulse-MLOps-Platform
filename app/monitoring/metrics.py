"""Prometheus metrics for DemandPulse AI."""

from __future__ import annotations

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

# ---- HTTP / API metrics ----
http_requests_total = Counter(
    "demandpulse_http_requests_total",
    "Total HTTP requests",
    ["method", "path", "status"],
)

http_request_duration_seconds = Histogram(
    "demandpulse_http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "path"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
)

# ---- Forecasting metrics ----
forecast_requests_total = Counter(
    "demandpulse_forecast_requests_total",
    "Total forecast requests",
    ["product", "model", "status"],
)

forecast_duration_seconds = Histogram(
    "demandpulse_forecast_duration_seconds",
    "Forecast endpoint duration in seconds",
    ["horizon"],
    buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0),
)

forecast_wape = Gauge(
    "demandpulse_forecast_wape",
    "Latest forecast WAPE (%) for a given product",
    ["product", "model"],
)

forecast_horizon_total = Gauge(
    "demandpulse_forecast_horizon_total",
    "Latest forecast total demand for a given horizon",
    ["product", "horizon"],
)

# ---- Upload / data metrics ----
uploads_total = Counter(
    "demandpulse_uploads_total",
    "Total upload requests",
    ["status"],
)

upload_rows_total = Counter(
    "demandpulse_upload_rows_total",
    "Total rows ingested via uploads",
)

# ---- Model / MLOps metrics ----
model_training_duration_seconds = Histogram(
    "demandpulse_model_training_duration_seconds",
    "Model training duration in seconds",
    ["model"],
    buckets=(0.5, 1.0, 5.0, 10.0, 30.0, 60.0, 120.0),
)

model_trainings_total = Counter(
    "demandpulse_model_trainings_total",
    "Total model training events",
    ["model", "source"],  # source = "cache_miss" | "cache_hit"
)

mlflow_runs_total = Counter(
    "demandpulse_mlflow_runs_total",
    "Total MLflow runs logged",
    ["best_model", "registered"],
)


def render_metrics() -> tuple[bytes, str]:
    """Return (payload, content_type) for the /metrics endpoint."""
    return generate_latest(), CONTENT_TYPE_LATEST
