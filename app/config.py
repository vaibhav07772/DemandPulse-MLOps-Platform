from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # App
    app_name: str = "DemandPulse AI"
    app_version: str = "1.2.0"
    environment: str = "development"
    log_level: str = "INFO"
    log_json: bool = False

    # Uploads
    max_upload_bytes: int = 50 * 1024 * 1024
    allowed_extensions: str = ".csv,.xlsx,.xls"

    # Forecasting
    allowed_horizons: str = "7,30,60,90"
    max_cached_models: int = 2

    # Paths
    base_dir: Path = Path(__file__).resolve().parent
    data_dir: Path = Path(__file__).resolve().parent.parent / "data"

    @property
    def upload_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def session_dir(self) -> Path:
        return self.data_dir / "sessions"

    @property
    def extension_set(self) -> set[str]:
        return {e.strip().lower() for e in self.allowed_extensions.split(",") if e.strip()}

    @property
    def horizon_set(self) -> set[int]:
        return {int(h.strip()) for h in self.allowed_horizons.split(",") if h.strip()}


@lru_cache
def get_settings() -> Settings:
    return Settings()