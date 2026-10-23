"""
Advanced Demand Forecasting Models
==================================
- XGBoost
- LightGBM
- CatBoost
- Ensemble (متوسط النماذج)
على بيانات أسبوعية بعد تنظيف Outliers
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

from src.models.train_weekly import (
    load_and_aggregate_weekly,
    add_features,
    clean_data,
)


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


def train_advanced():
    print("🚀 Advanced Models Training Pipeline\n")

    # ---- تحضير البيانات ----
    df = load_and_aggregate_weekly()
    df = add_features(df)
    df, feature_cols = clean_data(df)

    df = df.sort_values("week_start")
    split_idx = int(len(df) * 0.8)
    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]

    X_train = train[feature_cols]
    y_train = train["quantity_sold"]
    X_test = test[feature_cols]
    y_test = test["quantity_sold"]

    y_train_log = np.log1p(y_train)

    print(f"\nTrain: {len(train):,} | Test: {len(test):,}")
    print(f"Features: {len(feature_cols)}")

    results = []
    predictions = {}

    # ==================================================
    # 1) XGBoost
    # ==================================================
    print("\n⏳ تدريب XGBoost...")
    xgb = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
    )
    xgb.fit(X_train, y_train_log)
    pred_xgb = np.expm1(xgb.predict(X_test))
    predictions["xgb"] = pred_xgb
    results.append(evaluate(y_test, pred_xgb, "XGBoost"))

    # ==================================================
    # 2) LightGBM
    # ==================================================
    print("\n⏳ تدريب LightGBM...")
    lgb = LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=20,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    lgb.fit(X_train, y_train_log)
    pred_lgb = np.expm1(lgb.predict(X_test))
    predictions["lgb"] = pred_lgb
    results.append(evaluate(y_test, pred_lgb, "LightGBM"))

    # ==================================================
    # 3) CatBoost
    # ==================================================
    print("\n⏳ تدريب CatBoost...")
    cat = CatBoostRegressor(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=3,
        subsample=0.8,
        random_seed=42,
        verbose=0,
    )
    cat.fit(X_train, y_train_log)
    pred_cat = np.expm1(cat.predict(X_test))
    predictions["cat"] = pred_cat
    results.append(evaluate(y_test, pred_cat, "CatBoost"))

    # ==================================================
    # 4) Ensemble (متوسط النماذج الثلاثة)
    # ==================================================
    print("\n⏳ حساب Ensemble...")
    pred_ensemble = (pred_xgb + pred_lgb + pred_cat) / 3.0
    predictions["ensemble"] = pred_ensemble
    results.append(evaluate(y_test, pred_ensemble, "Ensemble (XGB+LGB+CAT)"))

    # ==================================================
    # ملخص المقارنة
    # ==================================================
    results_df = pd.DataFrame(results).sort_values("mae")
    print("\n" + "=" * 60)
    print("🏆 مقارنة النماذج (مرتبة حسب MAE)")
    print("=" * 60)
    print(results_df.to_string(index=False))
    print("=" * 60)

    best = results_df.iloc[0]
    print(f"\n🥇 أفضل نموذج: {best['model']}")
    print(f"   MAE={best['mae']:.4f} | RMSE={best['rmse']:.4f} | "
          f"MAPE={best['mape']:.2f}% | R²={best['r2']:.4f}")

    # ==================================================
    # حفظ النماذج
    # ==================================================
    os.makedirs("models", exist_ok=True)
    joblib.dump(xgb, "models/xgb_weekly.pkl")
    joblib.dump(lgb, "models/lgb_weekly.pkl")
    joblib.dump(cat, "models/cat_weekly.pkl")
    joblib.dump(feature_cols, "models/weekly_feature_columns.pkl")
    joblib.dump(results_df, "models/advanced_results.pkl")

    # حفظ أفضل نموذج كـ default
    best_name = best["model"]
    if "XGBoost" in best_name:
        joblib.dump(xgb, "models/best_weekly_model.pkl")
    elif "LightGBM" in best_name:
        joblib.dump(lgb, "models/best_weekly_model.pkl")
    elif "CatBoost" in best_name:
        joblib.dump(cat, "models/best_weekly_model.pkl")
    else:
        # Ensemble: نحفظ المتوسط منطقيًا عبر حفظ الثلاثة + config
        joblib.dump(
            {"type": "ensemble", "models": ["xgb", "lgb", "cat"]},
            "models/best_weekly_model.pkl",
        )

    print("\n💾 النماذج محفوظة في models/")
    print("   - xgb_weekly.pkl")
    print("   - lgb_weekly.pkl")
    print("   - cat_weekly.pkl")
    print("   - best_weekly_model.pkl")
    print("   - advanced_results.pkl")

    return results_df


if __name__ == "__main__":
    train_advanced()