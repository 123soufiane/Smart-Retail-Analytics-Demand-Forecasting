"""
Demand Forecasting - Weekly Level (Improved)
============================================
تجميع أسبوعي + تنظيف + log1p + XGBoost
"""

import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import joblib
from sqlalchemy import text
from src.database.connection import get_engine


def load_and_aggregate_weekly(
    sku_prefix: str | None = None, by_store: bool = False
) -> pd.DataFrame:
    print("⏳ تحميل البيانات وتجميعها أسبوعيًا...")
    engine = get_engine()

    where = ""
    params = {}
    if sku_prefix:
        where = "WHERE p.sku LIKE :prefix"
        params["prefix"] = f"{sku_prefix}%"

    store_select = "s.store_id," if by_store else ""
    store_group = "s.store_id, " if by_store else ""
    store_order = "store_id, " if by_store else ""

    query = f"""
    SELECT
        si.product_id,
        {store_select}
        DATE_TRUNC('week', s.sale_date)::date AS week_start,
        SUM(si.quantity) AS quantity_sold,
        AVG(si.unit_price) AS avg_unit_price,
        AVG(si.discount) AS avg_discount,
        COUNT(DISTINCT s.sale_id) AS num_transactions
    FROM sale_items si
    JOIN sales s ON si.sale_id = s.sale_id
    JOIN products p ON si.product_id = p.product_id
    {where}
    GROUP BY si.product_id, {store_group}DATE_TRUNC('week', s.sale_date)::date
    ORDER BY si.product_id, {store_order}week_start
    """
    df = pd.read_sql(text(query), engine, params=params)
    df["week_start"] = pd.to_datetime(df["week_start"])

    print(f"   صفوف أسبوعية: {len(df):,}")
    print(f"   منتجات: {df['product_id'].nunique():,}")
    print(f"   الفترة: {df['week_start'].min().date()} → {df['week_start'].max().date()}")
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    print("⏳ إنشاء الميزات الأسبوعية...")
    df = df.sort_values(["product_id", "week_start"]).copy()
    g = df.groupby("product_id")

    # Lag (بالأسابيع)
    df["lag_1"] = g["quantity_sold"].shift(1)
    df["lag_2"] = g["quantity_sold"].shift(2)
    df["lag_4"] = g["quantity_sold"].shift(4)
    df["lag_8"] = g["quantity_sold"].shift(8)

    # Rolling
    df["rolling_mean_4"] = g["quantity_sold"].transform(
        lambda x: x.shift(1).rolling(4, min_periods=1).mean()
    )
    df["rolling_mean_8"] = g["quantity_sold"].transform(
        lambda x: x.shift(1).rolling(8, min_periods=1).mean()
    )
    df["rolling_std_4"] = g["quantity_sold"].transform(
        lambda x: x.shift(1).rolling(4, min_periods=1).std()
    )

    # زمن
    df["week_of_year"] = df["week_start"].dt.isocalendar().week.astype(int)
    df["month"] = df["week_start"].dt.month
    df["quarter"] = df["week_start"].dt.quarter
    df["year"] = df["week_start"].dt.year

    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    print("⏳ تنظيف البيانات...")
    print(f"   قبل: {len(df):,}")

    # منتجات لها ≥ 8 أسابيع
    weeks_per_product = df.groupby("product_id")["week_start"].nunique()
    valid = weeks_per_product[weeks_per_product >= 8].index
    df = df[df["product_id"].isin(valid)]
    print(f"   بعد ≥8 أسابيع: {len(df):,} | منتجات: {df['product_id'].nunique():,}")

    # Outliers
    q99 = df["quantity_sold"].quantile(0.99)
    before = len(df)
    df = df[df["quantity_sold"] <= q99]
    print(f"   بعد outliers (>{q99:.0f}): {len(df):,} (حُذف {before - len(df):,})")

    feature_cols = [
        "lag_1", "lag_2", "lag_4", "lag_8",
        "rolling_mean_4", "rolling_mean_8", "rolling_std_4",
        "avg_unit_price", "avg_discount",
        "week_of_year", "month", "quarter", "year",
    ]
    for c in feature_cols:
        if c in df.columns:
            df[c] = df[c].fillna(0)

    return df, feature_cols


def train_weekly():
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

    print(f"\n   Train: {len(train):,} | Test: {len(test):,}")
    print(f"   Target range: {y_train.min():.0f} → {y_train.max():.0f}")

    print("⏳ تدريب XGBoost (Weekly)...")
    model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train_log)

    y_pred = np.clip(np.expm1(model.predict(X_test)), 0, None)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-6))) * 100
    r2 = r2_score(y_test, y_pred)

    print("\n" + "=" * 55)
    print("📊 نتائج التقييم — Weekly Model")
    print("=" * 55)
    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"MAPE : {mape:.2f}%")
    print(f"R²   : {r2:.4f}")
    print("=" * 55)

    importance = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    print("\n🔝 أهم الميزات:")
    print(importance.head(10).to_string(index=False))

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/xgboost_weekly_model.pkl")
    joblib.dump(feature_cols, "models/weekly_feature_columns.pkl")
    joblib.dump({"level": "weekly", "use_log1p": True}, "models/weekly_model_config.pkl")

    # حفظ الميزات الأسبوعية
    os.makedirs("data", exist_ok=True)
    df.to_parquet("data/features_weekly.parquet", index=False)
    print("\n💾 النموذج: models/xgboost_weekly_model.pkl")
    print("💾 الميزات: data/features_weekly.parquet")

    return model, {"mae": mae, "rmse": rmse, "mape": mape, "r2": r2}


if __name__ == "__main__":
    train_weekly()