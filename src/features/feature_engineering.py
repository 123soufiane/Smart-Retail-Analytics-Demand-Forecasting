import os
import pandas as pd
import numpy as np
from sqlalchemy import text
from src.database.connection import get_engine


def load_sales_data() -> pd.DataFrame:
    engine = get_engine()
    query = """
    SELECT
        s.sale_id, s.sale_date, s.store_id,
        si.product_id, si.quantity, si.unit_price, si.discount,
        p.product_name, p.category_id, p.cost_price,
        c.category_name,
        st.store_name, st.city AS store_city, st.country AS store_country
    FROM sales s
    JOIN sale_items si ON s.sale_id = si.sale_id
    JOIN products p ON si.product_id = p.product_id
    JOIN categories c ON p.category_id = c.category_id
    JOIN stores st ON s.store_id = st.store_id
    """
    df = pd.read_sql(query, engine)
    df['sale_date'] = pd.to_datetime(df['sale_date'])
    for col in ['product_name', 'category_name', 'store_name',
                'store_city', 'store_country']:
        df[col] = df[col].astype('category')
    return df


def create_time_and_revenue_features(df: pd.DataFrame) -> pd.DataFrame:
    # Modifies the frame in place to avoid doubling memory on 4M+ rows.
    df['year'] = df['sale_date'].dt.year
    df['month'] = df['sale_date'].dt.month
    df['day'] = df['sale_date'].dt.day
    df['day_of_week'] = df['sale_date'].dt.dayofweek
    df['quarter'] = df['sale_date'].dt.quarter
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

    df['revenue'] = df['quantity'] * df['unit_price'] * (1 - df['discount'] / 100)
    df['cost'] = df['quantity'] * df['cost_price']
    df['profit'] = df['revenue'] - df['cost']
    df['has_discount'] = (df['discount'] > 0).astype(int)
    return df


def create_daily_demand(df: pd.DataFrame) -> pd.DataFrame:
    daily = (
        df.groupby(['product_id', 'store_id', 'sale_date'])
        .agg(
            quantity_sold=('quantity', 'sum'),
            revenue=('revenue', 'sum'),
            num_transactions=('sale_id', 'nunique'),
            avg_discount=('discount', 'mean'),
            avg_unit_price=('unit_price', 'mean')
        )
        .reset_index()
    )

    product_info = df[['product_id', 'product_name', 'category_id', 'category_name']].drop_duplicates()
    store_info = df[['store_id', 'store_name', 'store_city', 'store_country']].drop_duplicates()

    daily = daily.merge(product_info, on='product_id', how='left')
    daily = daily.merge(store_info, on='store_id', how='left')
    return daily


def create_lag_features(daily: pd.DataFrame) -> pd.DataFrame:
    daily = daily.sort_values(['product_id', 'store_id', 'sale_date']).copy()
    group = daily.groupby(['product_id', 'store_id'])

    daily['lag_1'] = group['quantity_sold'].shift(1)
    daily['lag_7'] = group['quantity_sold'].shift(7)
    daily['lag_14'] = group['quantity_sold'].shift(14)
    daily['lag_30'] = group['quantity_sold'].shift(30)

    daily['rolling_mean_7'] = group['quantity_sold'].transform(
        lambda x: x.shift(1).rolling(7, min_periods=1).mean()
    )
    daily['rolling_mean_14'] = group['quantity_sold'].transform(
        lambda x: x.shift(1).rolling(14, min_periods=1).mean()
    )
    daily['rolling_std_7'] = group['quantity_sold'].transform(
        lambda x: x.shift(1).rolling(7, min_periods=1).std()
    )

    daily['year'] = daily['sale_date'].dt.year
    daily['month'] = daily['sale_date'].dt.month
    daily['day_of_week'] = daily['sale_date'].dt.dayofweek
    daily['is_weekend'] = daily['day_of_week'].isin([5, 6]).astype(int)
    daily['quarter'] = daily['sale_date'].dt.quarter
    return daily


def create_final_features() -> pd.DataFrame:
    print("⏳ تحميل بيانات المبيعات...")
    df = load_sales_data()
    print(f"   → {len(df):,} صف")

    print("⏳ إنشاء الميزات الزمنية والإيرادات...")
    df = create_time_and_revenue_features(df)

    print("⏳ تجميع الطلب اليومي...")
    daily = create_daily_demand(df)
    print(f"   → {len(daily):,} صف")
    del df

    print("⏳ إنشاء Lag & Rolling Features...")
    daily = create_lag_features(daily)

    print("✅ Feature Engineering اكتمل")
    return daily


def save_features(
    df: pd.DataFrame,
    path: str = "data/features_demand.parquet",
    write_csv: bool = False,
):
    os.makedirs("data", exist_ok=True)
    df.to_parquet(path, index=False)
    if write_csv:
        df.to_csv(path.replace(".parquet", ".csv"), index=False)
    print(f"💾 تم حفظ الميزات → {path}")


if __name__ == "__main__":
    features = create_final_features()
    save_features(features)