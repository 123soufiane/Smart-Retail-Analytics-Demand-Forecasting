"""
Weekly Demand Prediction — Strong Ensemble
==========================================
يستخدم XGB + LGB + CAT مع أوزان الـ Ensemble القوي
"""

import os
import argparse
import numpy as np
import pandas as pd
import joblib
from sqlalchemy import text
from src.database.connection import get_engine


def load_strong_models():
    models = {
        "xgb": joblib.load("models/xgb_strong.pkl"),
        "lgb": joblib.load("models/lgb_strong.pkl"),
        "cat": joblib.load("models/cat_strong.pkl"),
    }
    features = joblib.load("models/strong_feature_columns.pkl")
    weights = joblib.load("models/ensemble_weights.pkl")
    # weights قد تكون numpy array
    weights = np.array(weights, dtype=float)
    return models, features, weights


def get_history(product_id=None, top_n=None) -> pd.DataFrame:
    engine = get_engine()
    where = "WHERE si.product_id = :pid" if product_id is not None else ""
    params = {"pid": product_id} if product_id is not None else {}

    q = f"""
    SELECT
        si.product_id,
        p.product_name,
        p.sku,
        DATE_TRUNC('week', s.sale_date)::date AS week_start,
        SUM(si.quantity) AS quantity_sold,
        AVG(si.unit_price) AS avg_unit_price,
        AVG(COALESCE(si.discount, 0)) AS avg_discount
    FROM sale_items si
    JOIN sales s ON si.sale_id = s.sale_id
    JOIN products p ON si.product_id = p.product_id
    {where}
    GROUP BY 1,2,3,4
    ORDER BY si.product_id, week_start
    """
    df = pd.read_sql(text(q), engine, params=params)
    df["week_start"] = pd.to_datetime(df["week_start"])

    if top_n and product_id is None:
        top = df.groupby("product_id")["quantity_sold"].sum().nlargest(top_n).index
        df = df[df["product_id"].isin(top)]
    return df


def build_next_week_features(
    history: pd.DataFrame, feature_cols: list, group_cols=("product_id",)
) -> pd.DataFrame:
    history = history.sort_values(list(group_cols) + ["week_start"])
    rows = []

    for key, g in history.groupby(list(group_cols)):
        g = g.sort_values("week_start")
        if len(g) < 1:
            continue
        last = g.iloc[-1]
        next_week = last["week_start"] + pd.Timedelta(days=7)
        qty = g["quantity_sold"].values.astype(float)
        if isinstance(key, tuple):
            pid = key[0]
            store_id = key[1] if len(key) > 1 else None
        else:
            pid, store_id = key, None

        def lag(n):
            return float(qty[-n]) if len(qty) >= n else 0.0

        def roll_mean(n):
            w = qty[-n:] if len(qty) >= n else qty
            return float(np.mean(w)) if len(w) else 0.0

        def roll_std(n):
            w = qty[-n:] if len(qty) >= n else qty
            return float(np.std(w, ddof=1)) if len(w) > 1 else 0.0

        def roll_median(n):
            w = qty[-n:] if len(qty) >= n else qty
            return float(np.median(w)) if len(w) else 0.0

        lag1, lag4 = lag(1), lag(4)
        rm4, rm8 = roll_mean(4), roll_mean(8)

        feat = {
            "product_id": pid,
            "product_name": last.get("product_name", ""),
            "sku": last.get("sku", ""),
            "last_week": last["week_start"],
            "predict_week": next_week,
            "lag_1": lag1,
            "lag_2": lag(2),
            "lag_3": lag(3),
            "lag_4": lag4,
            "lag_8": lag(8),
            "lag_12": lag(12),
            "roll_mean_4": rm4,
            "roll_mean_8": rm8,
            "roll_mean_12": roll_mean(12),
            "roll_std_4": roll_std(4),
            "roll_std_8": roll_std(8),
            "roll_median_4": roll_median(4),
            "roll_median_8": roll_median(8),
            "momentum_4": lag1 / (rm4 + 1e-3),
            "momentum_8": lag1 / (rm8 + 1e-3),
            "lag_ratio_1_4": lag1 / (lag4 + 1e-3),
            "avg_unit_price": float(last["avg_unit_price"]),
            "avg_discount": float(last.get("avg_discount", 0) or 0),
            "week_of_year": int(next_week.isocalendar().week),
            "month": int(next_week.month),
            "quarter": int((next_week.month - 1) // 3 + 1),
            "year": int(next_week.year),
            "week_sin": np.sin(2 * np.pi * int(next_week.isocalendar().week) / 52),
            "week_cos": np.cos(2 * np.pi * int(next_week.isocalendar().week) / 52),
        }
        if store_id is not None:
            feat["store_id"] = int(store_id)
        rows.append(feat)

    df = pd.DataFrame(rows)
    for c in feature_cols:
        if c not in df.columns:
            df[c] = 0.0
        df[c] = df[c].fillna(0)
    return df


def predict_weighted(models, weights, X):
    order = ["xgb", "lgb", "cat"]
    parts = []
    for i, name in enumerate(order):
        p = np.expm1(models[name].predict(X))
        parts.append(weights[i] * p)
    return np.clip(np.sum(parts, axis=0), 0, None)


def predict_product(product_id: int):
    models, features, weights = load_strong_models()
    hist = get_history(product_id=product_id)
    if hist.empty:
        print(f"❌ لا بيانات للمنتج {product_id}")
        return
    feat = build_next_week_features(hist, features)
    pred = predict_weighted(models, weights, feat[features])[0]
    r = feat.iloc[0]
    print("\n" + "=" * 55)
    print("🔮 تنبؤ أسبوعي (Strong Ensemble)")
    print("=" * 55)
    print(f"Product ID    : {int(r['product_id'])}")
    print(f"Name          : {str(r['product_name'])[:55]}")
    print(f"SKU           : {r['sku']}")
    print(f"Last week     : {r['last_week'].date()}")
    print(f"Predict week  : {r['predict_week'].date()}")
    print(f"Last quantity : {r['lag_1']:.1f}")
    print(f"Predicted qty : {pred:.1f}")
    print("=" * 55)


def predict_top(n: int = 10):
    models, features, weights = load_strong_models()
    hist = get_history(top_n=n)
    feat = build_next_week_features(hist, features)
    preds = predict_weighted(models, weights, feat[features])
    feat["predicted_quantity"] = np.round(preds, 2)
    feat["last_quantity"] = feat["lag_1"].round(2)
    out = feat[
        ["product_id", "product_name", "sku", "last_week", "predict_week",
         "last_quantity", "predicted_quantity"]
    ].sort_values("predicted_quantity", ascending=False)

    print("\n" + "=" * 70)
    print(f"🔮 Strong Ensemble — Top {n}")
    print("=" * 70)
    print(out.to_string(index=False))
    print("=" * 70)

    os.makedirs("data", exist_ok=True)
    path = "data/predictions_next_week_strong.csv"
    out.to_csv(path, index=False)
    print(f"\n💾 {path}")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--product-id", type=int, default=None)
    p.add_argument("--top", type=int, default=10)
    args = p.parse_args()
    if args.product_id is not None:
        predict_product(args.product_id)
    else:
        predict_top(args.top)


if __name__ == "__main__":
    main()