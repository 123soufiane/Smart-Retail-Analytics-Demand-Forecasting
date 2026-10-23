"""
Exploratory Data Analysis (EDA)
===============================
تحليل استكشافي لبيانات المشروع
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sqlalchemy import text
from src.database.connection import get_engine

pd.set_option('display.max_columns', None)
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

FIGURES_DIR = Path("reports/eda")


def _finish_plot(fig, name: str):
    """Save a figure to disk and close it.

    Saving is important on headless terminals, where ``plt.show()`` does not
    display anything and does not close the figure. Closing avoids the next
    chart being drawn on the previous chart's axes.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / f"{name}.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    print(f"🖼️  الشكل محفوظ: {path}")
    if "agg" not in plt.get_backend().lower():
        plt.show()
    plt.close(fig)


def get_engine_connection():
    return get_engine()


def overview_tables():
    """نظرة عامة على عدد الصفوف في كل جدول"""
    engine = get_engine_connection()
    tables = [
        'categories', 'suppliers', 'products', 'stores',
        'customers', 'inventory', 'sales', 'sale_items', 'returns'
    ]

    print("\n" + "="*50)
    print("📊 نظرة عامة على الجداول")
    print("="*50)

    rows = []
    with engine.connect() as conn:
        for table in tables:
            count = conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()
            rows.append({"Table": table, "Rows": count})
            print(f"{table:15} → {count:,}")

    return pd.DataFrame(rows)


def monthly_sales():
    """تحليل المبيعات الشهرية"""
    engine = get_engine_connection()
    query = """
    SELECT 
        DATE_TRUNC('month', sale_date) AS month,
        COUNT(*) AS num_sales,
        ROUND(SUM(total_amount)::numeric, 2) AS total_revenue
    FROM sales
    GROUP BY 1
    ORDER BY 1
    """
    df = pd.read_sql(query, engine)
    df['month'] = pd.to_datetime(df['month'])

    print("\n" + "="*50)
    print("📈 المبيعات الشهرية")
    print("="*50)
    print(df)

    fig, ax1 = plt.subplots()
    ax1.plot(df['month'], df['total_revenue'], color='tab:blue', marker='o', label='Revenue')
    ax1.set_ylabel('Total Revenue', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx()
    ax2.bar(df['month'], df['num_sales'], alpha=0.3, color='tab:orange', width=20, label='Sales Count')
    ax2.set_ylabel('Number of Sales', color='tab:orange')
    ax2.tick_params(axis='y', labelcolor='tab:orange')

    plt.title('Monthly Sales & Revenue')
    plt.xticks(rotation=45)
    plt.tight_layout()
    _finish_plot(fig, "monthly_sales")

    return df


def top_products(n=10):
    """أفضل المنتجات حسب الإيرادات"""
    engine = get_engine_connection()
    query = f"""
    SELECT 
        p.product_name,
        SUM(si.quantity) AS total_qty,
        ROUND(SUM(si.quantity * si.unit_price * (1 - si.discount/100))::numeric, 2) AS revenue
    FROM sale_items si
    JOIN products p ON si.product_id = p.product_id
    GROUP BY p.product_name
    ORDER BY revenue DESC
    LIMIT {n}
    """
    df = pd.read_sql(query, engine)

    print("\n" + "="*50)
    print(f"🏆 أفضل {n} منتجات حسب الإيرادات")
    print("="*50)
    print(df)

    fig, ax = plt.subplots()
    sns.barplot(data=df, x='revenue', y='product_name', hue='product_name',
                palette='viridis', legend=False, ax=ax)
    ax.set_title(f'Top {n} Products by Revenue')
    ax.set_xlabel('Revenue')
    ax.set_ylabel('')
    fig.tight_layout()
    _finish_plot(fig, "top_products")

    return df


def revenue_by_category():
    """الإيرادات حسب التصنيف"""
    engine = get_engine_connection()
    query = """
    SELECT 
        c.category_name,
        ROUND(SUM(si.quantity * si.unit_price * (1 - si.discount/100))::numeric, 2) AS revenue
    FROM sale_items si
    JOIN products p ON si.product_id = p.product_id
    JOIN categories c ON p.category_id = c.category_id
    GROUP BY c.category_name
    ORDER BY revenue DESC
    """
    df = pd.read_sql(query, engine)

    print("\n" + "="*50)
    print("📦 الإيرادات حسب التصنيف")
    print("="*50)
    print(df)

    fig, ax = plt.subplots()
    sns.barplot(data=df, x='revenue', y='category_name', hue='category_name',
                palette='magma', legend=False, ax=ax)
    ax.set_title('Revenue by Category')
    ax.set_xlabel('Revenue')
    ax.set_ylabel('')
    fig.tight_layout()
    _finish_plot(fig, "revenue_by_category")

    return df


def payment_methods():
    """تحليل طرق الدفع"""
    engine = get_engine_connection()
    query = """
    SELECT 
        payment_method,
        COUNT(*) AS num_sales,
        ROUND(SUM(total_amount)::numeric, 2) AS total_revenue
    FROM sales
    GROUP BY payment_method
    ORDER BY total_revenue DESC
    """
    df = pd.read_sql(query, engine)

    print("\n" + "="*50)
    print("💳 تحليل طرق الدفع")
    print("="*50)
    print(df)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].pie(df['num_sales'], labels=df['payment_method'], autopct='%1.1f%%')
    axes[0].set_title('Sales Count by Payment Method')
    axes[1].pie(df['total_revenue'], labels=df['payment_method'], autopct='%1.1f%%')
    axes[1].set_title('Revenue by Payment Method')
    fig.tight_layout()
    _finish_plot(fig, "payment_methods")

    return df


def returns_analysis():
    """تحليل المرتجعات"""
    engine = get_engine_connection()
    query = """
    SELECT 
        reason,
        COUNT(*) AS num_returns,
        SUM(quantity) AS total_qty
    FROM returns
    GROUP BY reason
    ORDER BY num_returns DESC
    """
    df = pd.read_sql(query, engine)

    print("\n" + "="*50)
    print("↩️ تحليل المرتجعات")
    print("="*50)
    print(df)

    if len(df) > 0:
        fig, ax = plt.subplots()
        sns.barplot(data=df, x='num_returns', y='reason', hue='reason',
                    palette='Reds_r', legend=False, ax=ax)
        ax.set_title('Returns by Reason')
        ax.set_xlabel('Number of Returns')
        ax.set_ylabel('')
        fig.tight_layout()
        _finish_plot(fig, "returns")

    return df


def summary():
    """ملخص سريع"""
    engine = get_engine_connection()

    print("\n" + "="*50)
    print("📌 ملخص سريع")
    print("="*50)

    with engine.connect() as conn:
        total_revenue = conn.execute(text("SELECT SUM(total_amount) FROM sales")).scalar() or 0
        avg_order = conn.execute(text("SELECT AVG(total_amount) FROM sales")).scalar() or 0
        total_returns = conn.execute(text("SELECT COUNT(*) FROM returns")).scalar() or 0
        total_sales = conn.execute(text("SELECT COUNT(*) FROM sales")).scalar() or 0

    print(f"إجمالي المبيعات      : {total_sales:,}")
    print(f"إجمالي الإيرادات     : ${total_revenue:,.2f}")
    print(f"متوسط قيمة الطلب    : ${avg_order:,.2f}")
    print(f"عدد المرتجعات       : {total_returns:,}")


def run_full_eda():
    """تشغيل كل تحليلات EDA"""
    print("\n🔍 بدء التحليل الاستكشافي (EDA)...")
    overview_tables()
    monthly_sales()
    top_products()
    revenue_by_category()
    payment_methods()
    returns_analysis()
    summary()
    print("\n✅ EDA اكتمل بنجاح")


if __name__ == "__main__":
    run_full_eda()