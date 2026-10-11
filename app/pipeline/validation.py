"""Time-series cross-validation splitter + schema validation entrypoint."""

from __future__ import annotations

import pandas as pd

from app.pipeline.schema import SchemaValidationError, validate_schema


def time_series_splits(n_samples: int, n_splits: int = 3, test_size: int | None = None):
    """
    Produce (train_end, test_start, test_end) tuples for walk-forward validation.
    Indices are positional (0-based) on the sorted series.
    """
    if n_samples < n_splits + 1:
        return []

    if test_size is None:
        test_size = max(1, n_samples // (n_splits + 1))

    splits = []
    for i in range(1, n_splits + 1):
        train_end = n_samples - (n_splits - i + 1) * test_size
        test_start = train_end
        test_end = min(train_end + test_size, n_samples)
        if train_end <= 0 or test_end <= test_start:
            continue
        splits.append((train_end, test_start, test_end))

    return splits


def validate_dataset(data: pd.DataFrame) -> pd.DataFrame:
    """
    Validate an uploaded dataset against the demand schema.

    Called by the upload endpoint before preprocessing.
    Returns validated DataFrame; raises SchemaValidationError on failure.
    """
    required = {"date", "quantity", "product", "store", "price", "discount"}
    missing = required.difference(data.columns)
    if missing:
        raise SchemaValidationError(f"Dataset is missing required columns: {sorted(missing)}")

    if data.empty:
        raise SchemaValidationError("The uploaded dataset contains no rows.")

    return validate_schema(data)
