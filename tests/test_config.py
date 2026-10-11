"""Tests for the application settings layer."""

from app.config import Settings, get_settings


def test_default_settings_load():
    settings = Settings()
    assert settings.app_name == "DemandPulse AI"
    assert settings.environment in {"development", "staging", "production"}
    assert settings.max_upload_bytes > 0


def test_extension_set_parsing():
    settings = Settings(allowed_extensions=".csv,.xlsx,.json")
    assert settings.extension_set == {".csv", ".xlsx", ".json"}


def test_horizon_set_parsing():
    settings = Settings(allowed_horizons="7,30,60")
    assert settings.horizon_set == {7, 30, 60}


def test_get_settings_is_cached():
    a = get_settings()
    b = get_settings()
    assert a is b


def test_paths_derived_from_data_dir():
    settings = Settings()
    assert settings.upload_dir.name == "uploads"
    assert settings.session_dir.name == "sessions"
