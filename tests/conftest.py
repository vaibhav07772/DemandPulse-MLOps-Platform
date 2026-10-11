"""Shared pytest fixtures for DemandPulse tests."""

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="session")
def sample_history() -> pd.DataFrame:
    """A small but realistic daily demand history for a single product/store."""
    rng = np.random.default_rng(42)
    dates = pd.date_range("2023-01-01", periods=60, freq="D")
    base = 100 + 20 * np.sin(np.arange(60) / 7)
    noise = rng.normal(0, 8, size=60)
    quantity = np.clip(base + noise, 0, None).round(2)
    return pd.DataFrame(
        {
            "date": dates,
            "quantity": quantity,
            "product": ["P0001"] * 60,
            "store": ["S01"] * 60,
            "price": [10.0] * 60,
            "discount": [0.0] * 60,
        }
    )


@pytest.fixture(scope="session")
def sample_dataset_csv(tmp_path_factory) -> str:
    """A CSV file with multiple products/stores, ready to upload via API."""
    rng = np.random.default_rng(7)
    rows = []
    for product in ("P0001", "P0002"):
        for store in ("S01", "S02"):
            dates = pd.date_range("2023-01-01", periods=40, freq="D")
            base = 80 + (10 if product == "P0002" else 0)
            qty = np.clip(base + rng.normal(0, 5, size=40), 0, None).round(2)
            for d, q in zip(dates, qty, strict=False):
                rows.append(
                    {
                        "date": d.strftime("%Y-%m-%d"),
                        "quantity": float(q),
                        "product": product,
                        "store": store,
                        "price": 12.5,
                        "discount": 0.0,
                    }
                )
    df = pd.DataFrame(rows)
    path = tmp_path_factory.mktemp("data") / "sample.csv"
    df.to_csv(path, index=False)
    return str(path)


@pytest.fixture()
def client(monkeypatch) -> TestClient:
    """FastAPI TestClient with MLflow disabled (fast tests)."""
    monkeypatch.setenv("MLFLOW_ENABLED", "false")
    # Clear cached settings so env var takes effect
    from app.config import get_settings

    get_settings.cache_clear()

    from app.main import app

    with TestClient(app) as c:
        yield c

    get_settings.cache_clear()
