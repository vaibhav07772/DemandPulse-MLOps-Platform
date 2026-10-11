"""Pandera schema for uploaded demand datasets.

Validates structure, dtypes, ranges, and null constraints.
Used by validation.validate_dataset() before preprocessing.
"""

from __future__ import annotations

import pandas as pd
import pandera as pa
from pandera import Check, Column, DataFrameSchema

demand_schema = DataFrameSchema(
    columns={
        "date": Column(
            pa.DateTime,
            nullable=False,
            coerce=True,
            description="Observation date (any parseable format)",
        ),
        "quantity": Column(
            pa.Float,
            nullable=False,
            coerce=True,
            checks=[
                Check.ge(0, error="quantity must be >= 0"),
                Check.le(1_000_000, error="quantity exceeds sane upper bound"),
            ],
            description="Units demanded on that date",
        ),
        "product": Column(
            pa.String,
            nullable=False,
            coerce=True,
            checks=Check.str_length(min_value=1, max_value=200),
        ),
        "store": Column(
            pa.String,
            nullable=False,
            coerce=True,
            checks=Check.str_length(min_value=1, max_value=200),
        ),
        "price": Column(
            pa.Float,
            nullable=False,
            coerce=True,
            checks=[
                Check.ge(0, error="price must be >= 0"),
                Check.le(1_000_000, error="price exceeds sane upper bound"),
            ],
        ),
        "discount": Column(
            pa.Float,
            nullable=False,
            coerce=True,
            checks=[
                Check.ge(0, error="discount must be >= 0"),
                Check.le(1, error="discount must be <= 1 (fraction)"),
            ],
        ),
    },
    strict=False,  # allow extra columns (adapter may add them)
    coerce=True,  # try to cast types
    ordered=False,
)


class SchemaValidationError(ValueError):
    """Raised when an uploaded dataset violates the demand schema."""


def validate_schema(data: pd.DataFrame) -> pd.DataFrame:
    """
    Validate a DataFrame against the demand schema.

    Returns the validated (and coerced) DataFrame on success.
    Raises SchemaValidationError with a readable message on failure.
    """
    try:
        validated = demand_schema.validate(data, lazy=True)
        return validated
    except pa.errors.SchemaErrors as exc:
        # Build a compact, human-readable error summary
        failure = exc.failure_cases
        if failure is None or failure.empty:
            raise SchemaValidationError(str(exc)) from exc

        problems = []
        for _, row in failure.head(5).iterrows():
            col = row.get("column", "?")
            check = row.get("check", "?")
            problems.append(f"  • column '{col}' failed: {check}")
        summary = "Dataset schema validation failed:\n" + "\n".join(problems)
        if len(failure) > 5:
            summary += f"\n  ... and {len(failure) - 5} more issues."
        raise SchemaValidationError(summary) from exc
