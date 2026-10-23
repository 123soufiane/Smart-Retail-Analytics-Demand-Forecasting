"""
Advanced Training v2
====================
- منتجات نشطة فقط
- log1p target
- sample weights حسب حجم الطلب
- XGBoost + LightGBM + CatBoost + Ensemble
"""

import os
import warnings
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import joblib

warnings.filterwarnings("ignore")

from src.models.train_weekly import load_and_aggregate_weekly, add_features


def evaluate(y_true, y_pred, name: str) -> dict:
    y_pred = np.clip(y_pred, 0, None)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-6))) * 100
    r2 = r2_score(y_true, y_pred)

    print(f"\n{'='*50}")
    print(f"📊 {name}")
    print(f"{'='*50}")
    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"MAPE : {mape:.2f}%")
    print(f"R²   : {r2:.4f}")

    return {"model": name, "mae": mae, "rmse": rmse, "mape": mape, "r2": r2}


def filter_active_products(df: pd.DataFrame) -> pd.DataFrame:
    """
    منتجات نشطة:
    - على الأقل 12 أسبوع مبيعات
    - إجمالي كمية >= 50
    - متوسط أسبوعي >= 3
    """
    print("⏳ فلترة المنتجات النشطة...")
    print(f"   قبل: {len(df):,} صف | {df['product_id'].nunique():,} منتج")

    stats = df.groupby("product_id").agg(
        n_weeks=("week_start", "nunique"),
        total_qty=("quantity_sold", "sum"),
        avg_qty=("quantity_sold", "mean"),
    )

    active = stats[
        (stats["n_weeks"] >= 12)
        & (stats["total_qty"] >= 50)
        & (stats["avg_qty"] >= 3)
    ].index

    df = df[df["product_id"].isin(active)].copy()
    print(f"   بعد: {len(df):,} صف | {df['product_id'].nunique():,} منتج نشط")
    return df


def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    q99 = df["quantity_sold"].quantile(0.99)
    before = len(df)
    df = df[df["quantity_sold"] <= q99].copy()
    print(f"   outliers (>{q99:.0f}): حُذف {before - len(df):,} صف")
    return df


def train_v2():
    print("🚀 Advanced Training v2 (Active Products + Weights)\n")

    df = load_and_aggregate_weekly()
    df = add_features(df)
    df = filter_active_products(df)
    df = remove_outliers(df)

    feature_cols = [
        "lag_1", "lag_2", "lag_4", "lag_8",
        "rolling_mean_4", "rolling_mean_8", "rolling_std_4",
        "avg_unit_price", "avg_discount",
        "week_of_year", "month", "quarter", "year",
    ]
    for c in feature_cols:
        if c in df.columns:
            df[c] = df[c].fillna(0)

    df = df.sort_values("week_start")
    split_idx = int(len(df) * 0.8)
    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]

    X_train = train[feature_cols]
    y_train = train["quantity_sold"]
    X_test = test[feature_cols]
    y_test = test["quantity_sold"]

    # ---- Sample Weights ----
    # وزن أعلى للكميات الأكبر (لكن بحد أقصى عشان ما يطغى الـ outliers)
    sample_weights = np.clip(np.log1p(y_train), 1.0, 6.0)

    y_train_log = np.log1p(y_train)

    print(f"\nTrain: {len(train):,} | Test: {len(test):,}")
    print(f"Target range: {y_train.min():.0f} → {y_train.max():.0f}")
    print(f"Weights range: {sample_weights.min():.2f} → {sample_weights.max():.2f}")

    results = []
    preds = {}

    # ---- XGBoost ----
    print("\n⏳ XGBoost...")
    xgb = XGBRegressor(
        n_estimators=600,
        learning_rate=0.04,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_lambda=1.5,
        random_state=42,
        n_jobs=-1,
    )
    xgb.fit(X_train, y_train_log, sample_weight=sample_weights)
    pred_xgb = np.expm1(xgb.predict(X_test))
    preds["xgb"] = pred_xgb
    results.append(evaluate(y_test, pred_xgb, "XGBoost + Weights"))

    # ---- LightGBM ----
    print("\n⏳ LightGBM...")
    lgb = LGBMRegressor(
        n_estimators=600,
        learning_rate=0.04,
        max_depth=6,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=15,
        reg_lambda=1.5,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    lgb.fit(X_train, y_train_log, sample_weight=sample_weights)
    pred_lgb = np.expm1(lgb.predict(X_test))
    preds["lgb"] = pred_lgb
    results.append(evaluate(y_test, pred_lgb, "LightGBM + Weights"))

    # ---- CatBoost ----
    print("\n⏳ CatBoost...")
    cat = CatBoostRegressor(
        iterations=600,
        learning_rate=0.04,
        depth=6,
        l2_leaf_reg=4,
        subsample=0.8,
        random_seed=42,
        verbose=0,
    )
    cat.fit(X_train, y_train_log, sample_weight=sample_weights)
    pred_cat = np.expm1(cat.predict(X_test))
    preds["cat"] = pred_cat
    results.append(evaluate(y_test, pred_cat, "CatBoost + Weights"))

    # ---- Ensemble ----
    print("\n⏳ Ensemble...")
    pred_ens = (pred_xgb + pred_lgb + pred_cat) / 3.0
    results.append(evaluate(y_test, pred_ens, "Ensemble v2"))

    # ---- Summary ----
    results_df = pd.DataFrame(results).sort_values("mae")
    print("\n" + "=" * 65)
    print("🏆 مقارنة النماذج v2 (Active + Weights)")
    print("=" * 65)
    print(results_df.to_string(index=False))
    print("=" * 65)

    best = results_df.iloc[0]
    print(f"\n🥇 الأفضل: {best['model']}")
    print(
        f"   MAE={best['mae']:.4f} | RMSE={best['rmse']:.4f} | "
        f"MAPE={best['mape']:.2f}% | R²={best['r2']:.4f}"
    )

    # ---- Save ----
    os.makedirs("models", exist_ok=True)
    joblib.dump(xgb, "models/xgb_weekly_v2.pkl")
    joblib.dump(lgb, "models/lgb_weekly_v2.pkl")
    joblib.dump(cat, "models/cat_weekly_v2.pkl")
    joblib.dump(feature_cols, "models/weekly_feature_columns_v2.pkl")
    joblib.dump(results_df, "models/advanced_results_v2.pkl")
    joblib.dump(
        {
            "level": "weekly",
            "use_log1p": True,
            "use_weights": True,
            "active_filter": True,
        },
        "models/model_config_v2.pkl",
    )

    # best model
    name = best["model"]
    if "XGBoost" in name:
        joblib.dump(xgb, "models/best_weekly_model_v2.pkl")
    elif "LightGBM" in name:
        joblib.dump(lgb, "models/best_weekly_model_v2.pkl")
    elif "CatBoost" in name:
        joblib.dump(cat, "models/best_weekly_model_v2.pkl")
    else:
        joblib.dump({"type": "ensemble", "models": ["xgb", "lgb", "cat"]}, "models/best_weekly_model_v2.pkl")

    print("\n💾 تم الحفظ في models/*_v2.pkl")
    return results_df


if __name__ == "__main__":
    train_v2()