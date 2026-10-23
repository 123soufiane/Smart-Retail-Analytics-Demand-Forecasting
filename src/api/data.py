"""Load and summarize forecast files produced by the prediction pipeline."""

from pathlib import Path
from typing import Literal

import pandas as pd

ForecastLevel = Literal["store", "product"]

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
FORECAST_FILES: dict[ForecastLevel, str] = {
    "store": "predictions_store_next_week.csv",
    "product": "predictions_next_week_strong.csv",
}
REQUIRED_COLUMNS: dict[ForecastLevel, set[str]] = {
    "store": {
        "store_id",
        "store_name",
        "product_id",
        "product_name",
        "sku",
        "last_week",
        "predict_week",
        "last_quantity",
        "predicted_quantity",
    },
    "product": {
        "product_id",
        "product_name",
        "sku",
        "last_week",
        "predict_week",
        "last_quantity",
        "predicted_quantity",
    },
}


def load_forecasts(
    level: ForecastLevel, data_dir: Path = DATA_DIR
) -> pd.DataFrame:
    """Read and validate one of the pipeline's forecast output files."""
    path = data_dir / FORECAST_FILES[level]
    if not path.is_file():
        raise FileNotFoundError(
            f"Forecast file not found: {path}. Run the prediction pipeline first."
        )

    forecasts = pd.read_csv(path)
    missing_columns = REQUIRED_COLUMNS[level] - set(forecasts.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Forecast file {path} is missing required columns: {missing}")
    if forecasts.empty:
        raise ValueError(f"Forecast file contains no forecasts: {path}")

    for column in ("last_quantity", "predicted_quantity"):
        forecasts[column] = pd.to_numeric(forecasts[column], errors="coerce")
        if forecasts[column].isna().any():
            raise ValueError(f"Forecast file {path} has invalid values in {column}.")

    for column in ("last_week", "predict_week"):
        forecasts[column] = pd.to_datetime(forecasts[column], errors="coerce")
        if forecasts[column].isna().any():
            raise ValueError(f"Forecast file {path} has invalid dates in {column}.")

    return forecasts.sort_values("predicted_quantity", ascending=False).reset_index(
        drop=True
    )


def summarize_forecasts(forecasts: pd.DataFrame, level: ForecastLevel) -> dict:
    """Return headline demand metrics for a set of forecast rows."""
    last_quantity = float(forecasts["last_quantity"].sum())
    predicted_quantity = float(forecasts["predicted_quantity"].sum())
    percent_change = (
        (predicted_quantity - last_quantity) / last_quantity * 100
        if last_quantity
        else None
    )

    summary = {
        "level": level,
        "forecast_count": int(len(forecasts)),
        "product_count": int(forecasts["product_id"].nunique()),
        "last_quantity": last_quantity,
        "predicted_quantity": predicted_quantity,
        "percent_change": percent_change,
        "last_week": forecasts["last_week"].max().date().isoformat(),
        "predict_week": forecasts["predict_week"].max().date().isoformat(),
    }
    if level == "store":
        summary["store_count"] = int(forecasts["store_id"].nunique())
    return summary
