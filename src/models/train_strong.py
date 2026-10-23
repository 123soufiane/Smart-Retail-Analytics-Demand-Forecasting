"""
Strong Weekly Demand Model
==========================
ميزات أغنى + قص Outliers في الميزات + Ensemble موزون
"""

import os
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import joblib

warnings.filterwarnings("ignore")

from src.models.train_weekly import load_and_aggregate_weekly


def evaluate(y_true, y_pred, name: str) -> dict:
    y_pred = np.clip(y_pred, 0, None)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    # MAPE آمن للكميات الصغيرة
    mask = y_true >= 5
    mape = (
        np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        if mask.sum() > 0
        else np.nan
    )
    r2 = r2_score(y_true, y_pred)
    print(f"\n{'='*52}")
    print(f"📊 {name}")
    print(f"{'='*52}")
    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"MAPE*: {mape:.2f}%  (فقط حيث quantity >= 5)")
    print(f"R²   : {r2:.4f}")
    return {"model": name, "mae": mae, "rmse": rmse, "mape": mape, "r2": r2}


def build_strong_features(df: pd.DataFrame, group_cols=("product_id",)) -> pd.DataFrame:
    print("⏳ بناء ميزات قوية...")
    df = df.sort_values(list(group_cols) + ["week_start"]).copy()
    g = df.groupby(list(group_cols))["quantity_sold"]

    # Lags
    for n in [1, 2, 3, 4, 8, 12]:
        df[f"lag_{n}"] = g.shift(n)

    # Rolling
    for n in [4, 8, 12]:
        df[f"roll_mean_{n}"] = g.transform(
            lambda x: x.shift(1).rolling(n, min_periods=1).mean()
        )
        df[f"roll_std_{n}"] = g.transform(
            lambda x: x.shift(1).rolling(n, min_periods=2).std()
        )
        df[f"roll_median_{n}"] = g.transform(
            lambda x: x.shift(1).rolling(n, min_periods=1).median()
        )

    # Momentum & ratios
    df["momentum_4"] = df["lag_1"] / (df["roll_mean_4"] + 1e-3)
    df["momentum_8"] = df["lag_1"] / (df["roll_mean_8"] + 1e-3)
    df["lag_ratio_1_4"] = df["lag_1"] / (df["lag_4"] + 1e-3)
    df["zero_streak"] = (
        df.groupby(list(group_cols))["quantity_sold"]
        .transform(lambda s: (s.shift(1).fillna(0) == 0).astype(int))
    )

    # Calendar
    df["week_of_year"] = df["week_start"].dt.isocalendar().week.astype(int)
    df["month"] = df["week_start"].dt.month
    df["quarter"] = df["week_start"].dt.quarter
    df["year"] = df["week_start"].dt.year
    df["week_sin"] = np.sin(2 * np.pi * df["week_of_year"] / 52)
    df["week_cos"] = np.cos(2 * np.pi * df["week_of_year"] / 52)

    return df


def clean_and_cap(
    df: pd.DataFrame, feature_cols: list, group_cols=("product_id",)
) -> pd.DataFrame:
    print("⏳ تنظيف + قص القيم المتطرفة...")
    print(f"   قبل: {len(df):,} | سلاسل: {df.groupby(list(group_cols)).ngroups:,}")

    # سلاسل نشطة باعتدال
    stats = df.groupby(list(group_cols)).agg(
        n_weeks=("week_start", "nunique"),
        total_qty=("quantity_sold", "sum"),
        avg_qty=("quantity_sold", "mean"),
    )
    active = stats[
        (stats["n_weeks"] >= 10)
        & (stats["total_qty"] >= 40)
        & (stats["avg_qty"] >= 2)
    ].index
    active_keys = stats.loc[active].reset_index()[list(group_cols)]
    df = df.merge(active_keys, on=list(group_cols), how="inner").copy()

    # قص الهدف عند 99%
    q99 = df["quantity_sold"].quantile(0.99)
    df = df[df["quantity_sold"] <= q99].copy()

    # قص الميزات الشاذة (lags/rolling) عند 99% لكل عمود
    for c in feature_cols:
        if c in df.columns and df[c].dtype != object:
            hi = df[c].quantile(0.99)
            lo = df[c].quantile(0.01)
            df[c] = df[c].clip(lo, hi)

    for c in feature_cols:
        if c in df.columns:
            df[c] = df[c].fillna(0)

    print(
        f"   بعد: {len(df):,} | سلاسل: {df.groupby(list(group_cols)).ngroups:,} "
        f"| cap qty≤{q99:.0f}"
    )
    return df


def get_feature_list(df: pd.DataFrame) -> list:
    cols = [
        "lag_1", "lag_2", "lag_3", "lag_4", "lag_8", "lag_12",
        "roll_mean_4", "roll_mean_8", "roll_mean_12",
        "roll_std_4", "roll_std_8",
        "roll_median_4", "roll_median_8",
        "momentum_4", "momentum_8", "lag_ratio_1_4",
        "avg_unit_price", "avg_discount",
        "week_of_year", "month", "quarter", "year",
        "week_sin", "week_cos",
    ]
    return [c for c in cols if c in df.columns]


def train_strong(sku_prefix: str | None = None, by_store: bool = False):
    print("🚀 Strong Model Training\n")

    group_cols = ["product_id"] + (["store_id"] if by_store else [])
    df = load_and_aggregate_weekly(sku_prefix=sku_prefix, by_store=by_store)
    df = build_strong_features(df, group_cols=tuple(group_cols))
    feature_cols = get_feature_list(df)
    df = clean_and_cap(df, feature_cols, group_cols=tuple(group_cols))

    df = df.sort_values("week_start")
    split = int(len(df) * 0.8)
    train, test = df.iloc[:split], df.iloc[split:]

    X_train, y_train = train[feature_cols], train["quantity_sold"]
    X_test, y_test = test[feature_cols], test["quantity_sold"]
    y_log = np.log1p(y_train)
    # أوزان معتدلة
    sw = np.clip(np.sqrt(y_train + 1), 1.0, 8.0)

    print(f"Train: {len(train):,} | Test: {len(test):,} | Features: {len(feature_cols)}")

    results, preds = [], {}

    # ---- XGBoost ----
    print("\n⏳ XGBoost Strong...")
    xgb = XGBRegressor(
        n_estimators=800,
        learning_rate=0.03,
        max_depth=7,
        min_child_weight=8,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.2,
        reg_lambda=2.0,
        random_state=42,
        n_jobs=-1,
    )
    xgb.fit(X_train, y_log, sample_weight=sw)
    p_xgb = np.clip(np.expm1(xgb.predict(X_test)), 0, None)
    preds["xgb"] = p_xgb
    results.append(evaluate(y_test, p_xgb, "XGBoost Strong"))

    # ---- LightGBM ----
    print("\n⏳ LightGBM Strong...")
    lgb = LGBMRegressor(
        n_estimators=800,
        learning_rate=0.03,
        max_depth=7,
        num_leaves=48,
        min_child_samples=25,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.2,
        reg_lambda=2.0,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    lgb.fit(X_train, y_log, sample_weight=sw)
    p_lgb = np.clip(np.expm1(lgb.predict(X_test)), 0, None)
    preds["lgb"] = p_lgb
    results.append(evaluate(y_test, p_lgb, "LightGBM Strong"))

    # ---- CatBoost ----
    print("\n⏳ CatBoost Strong...")
    cat = CatBoostRegressor(
        iterations=800,
        learning_rate=0.03,
        depth=7,
        l2_leaf_reg=5,
        subsample=0.85,
        random_seed=42,
        verbose=0,
    )
    cat.fit(X_train, y_log, sample_weight=sw)
    p_cat = np.clip(np.expm1(cat.predict(X_test)), 0, None)
    preds["cat"] = p_cat
    results.append(evaluate(y_test, p_cat, "CatBoost Strong"))

    # ---- Weighted Ensemble (وزن عكسي لـ MAE) ----
    print("\n⏳ Weighted Ensemble...")
    maes = np.array([
        results[0]["mae"], results[1]["mae"], results[2]["mae"]
    ], dtype=float)
    weights = 1.0 / (maes + 1e-6)
    weights = weights / weights.sum()
    print(f"   الأوزان → XGB={weights[0]:.3f}, LGB={weights[1]:.3f}, CAT={weights[2]:.3f}")

    p_ens = weights[0] * p_xgb + weights[1] * p_lgb + weights[2] * p_cat
    results.append(evaluate(y_test, p_ens, "Weighted Ensemble Strong"))

    # ---- Summary ----
    res = pd.DataFrame(results).sort_values("mae")
    print("\n" + "=" * 64)
    print("🏆 مقارنة النماذج القوية")
    print("=" * 64)
    print(res.to_string(index=False))
    print("=" * 64)
    best = res.iloc[0]
    print(f"\n🥇 الأفضل: {best['model']} | MAE={best['mae']:.4f} | R²={best['r2']:.4f}")

    # ---- Save ----
    os.makedirs("models", exist_ok=True)
    if by_store:
        names = {
            "xgb": "store_xgb.pkl",
            "lgb": "store_lgb.pkl",
            "cat": "store_cat.pkl",
            "features": "store_feature_columns.pkl",
            "weights": "store_ensemble_weights.pkl",
            "results": "store_results.pkl",
            "config": "store_config.pkl",
        }
    else:
        names = {
            "xgb": "xgb_strong.pkl",
            "lgb": "lgb_strong.pkl",
            "cat": "cat_strong.pkl",
            "features": "strong_feature_columns.pkl",
            "weights": "ensemble_weights.pkl",
            "results": "strong_results.pkl",
            "config": "strong_config.pkl",
        }

    joblib.dump(xgb, f"models/{names['xgb']}")
    joblib.dump(lgb, f"models/{names['lgb']}")
    joblib.dump(cat, f"models/{names['cat']}")
    joblib.dump(feature_cols, f"models/{names['features']}")
    joblib.dump(weights, f"models/{names['weights']}")
    joblib.dump(res, f"models/{names['results']}")
    joblib.dump(
        {
            "use_log1p": True,
            "level": "weekly",
            "by_store": by_store,
            "sku_prefix": sku_prefix,
            "ensemble_weights": weights.tolist(),
            "features": feature_cols,
        },
        f"models/{names['config']}",
    )
    print(f"\n💾 حُفظ في models/ ({names['xgb']}, {names['lgb']}, {names['cat']})")
    return res


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("sku_prefix", nargs="?", default=None,
                        help="e.g. SITEM- to train on the Store Item dataset")
    parser.add_argument("--by-store", action="store_true",
                        help="train one model per product/store series")
    args = parser.parse_args()

    train_strong(args.sku_prefix, by_store=args.by_store)