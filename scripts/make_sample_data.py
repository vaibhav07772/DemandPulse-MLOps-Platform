"""Generate a small sample demand dataset for demos and tests.

Usage:
    python scripts/make_sample_data.py
Writes to: data/sample/demand_sample.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def make_sample(rows_per_series: int = 180, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []

    products = ["P0001", "P0002", "P0003"]
    stores = ["S01", "S02"]

    for product in products:
        for store in stores:
            dates = pd.date_range("2023-01-01", periods=rows_per_series, freq="D")
            base = 80 + (20 if product == "P0002" else 0)
            trend = np.linspace(0, 15, rows_per_series)
            weekly = 15 * np.sin(np.arange(rows_per_series) / 7 * 2 * np.pi)
            noise = rng.normal(0, 7, size=rows_per_series)
            qty = np.clip(base + trend + weekly + noise, 0, None).round(2)

            price = np.full(rows_per_series, 12.5)
            discount = np.where(rng.random(rows_per_series) < 0.1, 0.15, 0.0)

            for d, q, p, disc in zip(dates, qty, price, discount, strict=False):
                rows.append(
                    {
                        "date": d.strftime("%Y-%m-%d"),
                        "quantity": float(q),
                        "product": product,
                        "store": store,
                        "price": float(p),
                        "discount": float(disc),
                    }
                )

    return pd.DataFrame(rows)


def main() -> None:
    out_dir = Path("data/sample")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "demand_sample.csv"

    df = make_sample()
    df.to_csv(out_path, index=False)

    print(f"[sample] wrote {len(df)} rows → {out_path}")
    print(f"[sample] products={df['product'].nunique()} stores={df['store'].nunique()}")


if __name__ == "__main__":
    main()
