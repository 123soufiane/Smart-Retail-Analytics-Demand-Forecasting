"""
Model Evaluation Report
Loads the trained XGBoost model + saved features and prints accuracy metrics,
feature importance, and a quick residual analysis.
"""

import os
import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
FEATURES_PATH = os.path.join(PROJECT_ROOT, "data", "features_demand.parquet")

TARGET = "quantity_sold"
NON_FEATURE_COLUMNS = [
    TARGET,
    "sale_date",
    "product_name",
    "category_name",
    "store_name",
    "store_city",
    "store_country",
]


def load_model_and_features():
    model = joblib.load(os.path.join(MODELS_DIR, "xgboost_demand_model.pkl"))
    feature_cols = joblib.load(os.path.join(MODELS_DIR, "feature_columns.pkl"))
    df = pd.read_parquet(FEATURES_PATH)
    return model, feature_cols, df


def prepare(df, feature_cols):
    df = df.copy()
    df = df.dropna(subset=[TARGET])
    X = df[feature_cols].copy()
    y = df[TARGET].copy()
    for col in X.select_dtypes(include=["object", "category"]).columns:
        X[col] = X[col].astype("category").cat.codes
    X = X.fillna(0)
    return X, y, df


if __name__ == "__main__":
    model, feature_cols, df = load_model_and_features()
    X, y, df_clean = prepare(df, feature_cols)

    # Time-ordered split (same as training)
    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    y_pred = model.predict(X_test)
    y_pred = np.clip(y_pred, 0, None)

    y_test_np = y_test.to_numpy()
    mae = float(np.mean(np.abs(y_test_np - y_pred)))
    rmse = float(np.sqrt(np.mean((y_test_np - y_pred) ** 2)))
    mape = float(
        np.mean(np.abs((y_test_np - y_pred) / np.where(y_test_np == 0, 1, y_test_np))) * 100
    )
    ss_res = float(np.sum((y_test_np - y_pred) ** 2))
    ss_tot = float(np.sum((y_test_np - np.mean(y_test_np)) ** 2))
    r2 = float(1 - ss_res / ss_tot) if ss_tot != 0 else 0.0

    print("=" * 60)
    print("📊 DEMAND FORECASTING MODEL — EVALUATION REPORT")
    print("=" * 60)
    print(f"\n🗂  Test samples : {len(y_test):,}")
    print(f"🔢 Features used : {len(feature_cols)}")
    print("\n📈 Accuracy Metrics (Test Set):")
    print(f"   MAE  (Mean Absolute Error) : {mae:.4f}")
    print(f"   RMSE (Root Mean Squared Error): {rmse:.4f}")
    print(f"   MAPE (Mean Absolute Percentage Error): {mape:.2f}%")
    print(f"   R²   (Coefficient of Determination): {r2:.4f}")

    print("\n🎯 Target stats (Test Set):")
    print(f"   Actual  → mean: {y_test_np.mean():.2f}, min: {y_test_np.min()}, max: {y_test_np.max()}")
    print(f"   Predicted → mean: {y_pred.mean():.2f}, min: {y_pred.min():.2f}, max: {y_pred.max():.2f}")

    # Feature importance
    imp = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)

    print("\n🔝 Top 10 Important Features:")
    print(imp.head(10).to_string(index=False))

    # Residuals
    residuals = y_test_np - y_pred
    print("\n📉 Residuals (Test Set):")
    print(f"   Mean residual : {residuals.mean():.4f}")
    print(f"   Std residual  : {residuals.std():.4f}")
    print(f"   Min residual  : {residuals.min():.4f}")
    print(f"   Max residual  : {residuals.max():.4f}")

    print("\n" + "=" * 60)
