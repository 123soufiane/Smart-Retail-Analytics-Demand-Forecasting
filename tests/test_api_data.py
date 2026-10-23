"""Tests for loading and summarizing generated forecast files."""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.api.data import load_forecasts, summarize_forecasts


class ForecastDataTests(unittest.TestCase):
    def test_loads_and_sorts_store_forecasts(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions_store_next_week.csv"
            pd.DataFrame(
                [
                    {
                        "store_id": 1,
                        "store_name": "Store 1",
                        "product_id": 10,
                        "product_name": "Item 10",
                        "sku": "SKU-10",
                        "last_week": "2025-01-06",
                        "predict_week": "2025-01-13",
                        "last_quantity": 8,
                        "predicted_quantity": 12,
                    },
                    {
                        "store_id": 2,
                        "store_name": "Store 2",
                        "product_id": 11,
                        "product_name": "Item 11",
                        "sku": "SKU-11",
                        "last_week": "2025-01-06",
                        "predict_week": "2025-01-13",
                        "last_quantity": 10,
                        "predicted_quantity": 9,
                    },
                ]
            ).to_csv(path, index=False)

            forecasts = load_forecasts("store", Path(directory))

            self.assertEqual(forecasts.iloc[0]["product_id"], 10)
            self.assertEqual(
                forecasts.iloc[0]["predict_week"].date().isoformat(), "2025-01-13"
            )
            summary = summarize_forecasts(forecasts, "store")
            self.assertEqual(summary["forecast_count"], 2)
            self.assertEqual(summary["store_count"], 2)
            self.assertEqual(summary["predicted_quantity"], 21)
            self.assertAlmostEqual(summary["percent_change"], 16.666667, places=5)

    def test_missing_forecast_file_has_actionable_error(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, "Run the prediction pipeline"):
                load_forecasts("product", Path(directory))

    def test_rejects_missing_required_columns(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions_next_week_strong.csv"
            pd.DataFrame([{"product_id": 1}]).to_csv(path, index=False)

            with self.assertRaisesRegex(ValueError, "missing required columns"):
                load_forecasts("product", Path(directory))


if __name__ == "__main__":
    unittest.main()
