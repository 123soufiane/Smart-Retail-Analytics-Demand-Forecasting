import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import joblib


FEATURE_COLS = [
    "lag_1", "lag_7", "lag_14", "lag_30",
    "rolling_mean_7", "rolling_mean_14", "rolling_std_7",
    "avg_discount", "avg_unit_price",
    "day_of_week", "is_weekend", "month", "quarter", "year",
]

TARGET_COL = "quantity_sold"


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """تنظيف البيانات قبل التدريب"""
    df = df.copy()

    print("⏳ تنظيف البيانات...")
    print(f"   قبل التنظيف: {len(df):,} صف")

    # 1) منتجات لها على الأقل 30 يوم مبيعات
    product_days = df.groupby("product_id")["sale_date"].nunique()
    valid_products = product_days[product_days >= 30].index
    df = df[df["product_id"].isin(valid_products)]
    print(f"   بعد فلتر ≥30 يوم: {len(df):,} صف | منتجات: {df['product_id'].nunique():,}")

    # 2) إزالة القيم المتطرفة (فوق الـ 99 percentile)
    q99 = df[TARGET_COL].quantile(0.99)
    before = len(df)
    df = df[df[TARGET_COL] <= q99]
    print(f"   بعد إزالة Outliers (>{q99:.0f}): {len(df):,} صف (حُذف {before - len(df):,})")

    # 3) ملء القيم الناقصة في الميزات
    available = [c for c in FEATURE_COLS if c in df.columns]
    df[available] = df[available].fillna(0)

    return df


def train_model(df: pd.DataFrame):
    print("⏳ تجهيز بيانات التدريب (محسّن لـ UCI)...")

    df = prepare_data(df)
    df = df.sort_values("sale_date")

    available_features = [c for c in FEATURE_COLS if c in df.columns]

    # تقسيم زمني 80/20
    split_idx = int(len(df) * 0.8)
    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]

    X_train = train[available_features]
    y_train = train[TARGET_COL]
    X_test = test[available_features]
    y_test = test[TARGET_COL]

    # تحويل لوغاريتمي للهدف
    y_train_log = np.log1p(y_train)
    y_test_log = np.log1p(y_test)

    print(f"   Train: {len(train):,} | Test: {len(test):,}")
    print(f"   Features: {available_features}")
    print(f"   Target range (train): {y_train.min():.0f} → {y_train.max():.0f}")

    print("⏳ تدريب XGBoost على log1p(quantity)...")
    model = XGBRegressor(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train_log)

    # التنبؤ وإرجاع القيم الأصلية
    y_pred_log = model.predict(X_test)
    y_pred = np.expm1(y_pred_log)
    y_pred = np.clip(y_pred, 0, None)  # لا قيم سالبة

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-6))) * 100
    r2 = r2_score(y_test, y_pred)

    print("\n" + "=" * 55)
    print("📊 نتائج التقييم (بعد التحسين)")
    print("=" * 55)
    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"MAPE : {mape:.2f}%")
    print(f"R²   : {r2:.4f}")
    print("=" * 55)

    importance = pd.DataFrame({
        "feature": available_features,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    print("\n🔝 أهم الميزات:")
    print(importance.head(10).to_string(index=False))

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/xgboost_demand_model.pkl")
    joblib.dump(available_features, "models/feature_columns.pkl")
    joblib.dump({"use_log1p": True}, "models/model_config.pkl")
    print("\n💾 النموذج محفوظ في models/")

    return model, {
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "r2": r2,
    }


if __name__ == "__main__":
    df = pd.read_parquet("data/features_demand.parquet")
    train_model(df)