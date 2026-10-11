"""Smoke tests for the forecasting pipeline."""

import numpy as np
import pandas as pd
import pytest

from app.pipeline.forecasting import (
    _history_key,
    infer_frequency,
    recursive_forecast,
)


def test_infer_frequency_daily(sample_history):
    freq = infer_frequency(sample_history["date"])
    assert "Day" in str(freq) or "D" in str(freq)


def test_infer_frequency_handles_short_series():
    freq = infer_frequency(pd.Series(pd.to_datetime(["2023-01-01", "2023-01-02"])))
    assert freq is not None


def test_history_key_is_stable(sample_history):
    key1 = _history_key(sample_history)
    key2 = _history_key(sample_history.copy())
    assert key1 == key2
    assert len(key1) == 64


def test_history_key_changes_with_data(sample_history):
    key1 = _history_key(sample_history)
    modified = sample_history.copy()
    modified.loc[0, "quantity"] += 999
    key2 = _history_key(modified)
    assert key1 != key2


def test_recursive_forecast_runs_end_to_end(sample_history, monkeypatch):
    # Keep MLflow out of the test
    monkeypatch.setenv("MLFLOW_ENABLED", "false")
    from app.config import get_settings

    get_settings.cache_clear()
    from app.pipeline import forecasting

    # Clear module-level cache between tests
    forecasting._TRAINED_CACHE.clear()

    result = recursive_forecast(sample_history, horizon=7)

    assert result["model"] in {"Random Forest", "XGBoost", "LightGBM"}
    assert result["rows_used"] == len(sample_history)
    forecast = result["forecast"]
    assert isinstance(forecast, pd.DataFrame)
    assert len(forecast) == 7
    assert list(forecast.columns) == ["date", "forecast", "lower", "upper"]
    assert (forecast["forecast"] >= 0).all()
    assert (forecast["lower"] >= 0).all()
    assert (forecast["upper"] >= forecast["forecast"]).all()


def test_recursive_forecast_rejects_bad_horizon(sample_history):
    with pytest.raises(ValueError, match="horizon"):
        recursive_forecast(sample_history, horizon=11)
