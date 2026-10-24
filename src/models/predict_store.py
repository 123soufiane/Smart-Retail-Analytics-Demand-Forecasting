"""Per-store weekly demand prediction for the Store Item dataset.

Uses the store-level ensemble produced by:

    python -m src.models.train_strong SITEM- --by-store

The output has one row per product/store series and is written to
``data/predictions_store_next_week.csv``.
"""

import argparse
import os

import numpy as np
import pandas as pd
import joblib
from sqlalchemy import text

from src.database.connection import get_engine
from src.display_names import display_product_name, display_store_name
from src.models.predict import build_next_week_features
from src.models.train_weekly import load_and_aggregate_weekly


def load_store_models():
    models = {
        name: joblib.load(f"models/store_{name}.pkl")
        for name in ("xgb", "lgb", "cat")
    }
    features = joblib.load("models/store_feature_columns.pkl")
    weights = np.array(joblib.load("models/store_ensemble_weights.pkl"), dtype=float)
    return models, features, weights


def predict_weighted(models, weights, X):
    order = ["xgb", "lgb", "cat"]
    parts = [
        weights[i] * np.expm1(models[name].predict(X))
        for i, name in enumerate(order)
    ]
    return np.clip(np.sum(parts, axis=0), 0, None)


def reference_tables():
    engine = get_engine()
    products = pd.read_sql(
        text("SELECT product_id, product_name, sku FROM products"), engine
    )
    stores = pd.read_sql(
        text("SELECT store_id, store_name FROM stores"), engine
    )
    return products, stores


def predict_all() -> pd.DataFrame:
    models, features, weights = load_store_models()

    history = load_and_aggregate_weekly(sku_prefix="SITEM-", by_store=True)
    feat = build_next_week_features(
        history, features, group_cols=("product_id", "store_id")
    )
    feat["predicted_quantity"] = np.round(
        predict_weighted(models, weights, feat[features]), 2
    )
    feat["last_quantity"] = feat["lag_1"].round(2)

    products, stores = reference_tables()
    drop_empty = [c for c in ("product_name", "sku") if c in feat.columns]
    feat = feat.drop(columns=drop_empty)
    out = feat.merge(products, on="product_id", how="left").merge(
        stores, on="store_id", how="left"
    )
    out["product_name"] = [
        display_product_name(sku, name)
        for sku, name in zip(out["sku"], out["product_name"])
    ]
    out["store_name"] = out["store_name"].map(display_store_name)
    out = out[
        [
            "store_id",
            "store_name",
            "product_id",
            "product_name",
            "sku",
            "last_week",
            "predict_week",
            "last_quantity",
            "predicted_quantity",
        ]
    ].sort_values("predicted_quantity", ascending=False)

    os.makedirs("data", exist_ok=True)
    path = "data/predictions_store_next_week.csv"
    out.to_csv(path, index=False)
    print(f"💾 {path} ({len(out):,} صف)")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top", type=int, default=15)
    args = parser.parse_args()

    out = predict_all()
    print("\n" + "=" * 95)
    print(f"🔮 Per-store predictions — top {args.top}")
    print("=" * 95)
    print(out.head(args.top).to_string(index=False))
    print("=" * 95)


if __name__ == "__main__":
    main()
