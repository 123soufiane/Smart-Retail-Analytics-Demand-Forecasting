"""
إدخال سريع ومضمون لمبيعات UCI
باستخدام Bulk Insert (to_sql)
"""

import pandas as pd
from sqlalchemy import text
from src.database.connection import get_engine


def count(engine, table):
    with engine.connect() as conn:
        return conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()


def ensure_invoice_column(engine):
    """إضافة عمود invoice_no لو مش موجود"""
    with engine.begin() as conn:
        conn.execute(text("""
            ALTER TABLE sales
            ADD COLUMN IF NOT EXISTS invoice_no VARCHAR(50)
        """))
    print("✅ عمود invoice_no جاهز")


def load_clean(path="data/raw/OnlineRetail.xlsx"):
    print("⏳ قراءة وتنظيف البيانات...")
    df = pd.read_excel(path, engine="openpyxl")

    df = df.dropna(subset=["InvoiceNo", "StockCode", "Quantity", "InvoiceDate", "UnitPrice"])
    df["InvoiceNo"] = df["InvoiceNo"].astype(str)
    df = df[~df["InvoiceNo"].str.startswith("C")]
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["StockCode"] = df["StockCode"].astype(str).str.strip()
    df["CustomerID"] = df["CustomerID"].fillna(-1).astype(int)

    print(f"   الصفوف: {len(df):,}")
    return df


def run():
    engine = get_engine()

    # تخطي لو المبيعات موجودة
    if count(engine, "sales") > 0:
        print(f"⏭️  sales موجودة ({count(engine, 'sales'):,}) → تخطي")
        print(f"   sale_items → {count(engine, 'sale_items'):,}")
        return

    ensure_invoice_column(engine)
    df = load_clean()

    # ---- خرائط ----
    print("⏳ تحضير الخرائط...")
    with engine.connect() as conn:
        store_id = conn.execute(text("SELECT store_id FROM stores LIMIT 1")).scalar()
        sku_map = dict(conn.execute(text("SELECT sku, product_id FROM products")).fetchall())
        cust_rows = conn.execute(text(
            "SELECT last_name, customer_id FROM customers WHERE first_name = 'Customer'"
        )).fetchall()
        cust_map = {}
        for last, cid in cust_rows:
            try:
                cust_map[int(last)] = cid
            except Exception:
                pass

    df["product_id"] = df["StockCode"].map(sku_map)
    df = df.dropna(subset=["product_id"])
    df["product_id"] = df["product_id"].astype(int)
    df["customer_id"] = df["CustomerID"].map(cust_map)

    # ---- 1) sales ----
    print("⏳ إدخال sales...")
    sales = (
        df.groupby("InvoiceNo", as_index=False)
        .agg(
            customer_id=("customer_id", "first"),
            sale_date=("InvoiceDate", "first"),
            total_amount=("Quantity", lambda x: float((x * df.loc[x.index, "UnitPrice"]).sum())),
        )
    )
    sales["store_id"] = int(store_id)
    sales["payment_method"] = "Online"
    sales = sales.rename(columns={"InvoiceNo": "invoice_no"})

    sales_out = sales[[
        "customer_id", "store_id", "sale_date",
        "payment_method", "total_amount", "invoice_no"
    ]]

    sales_out.to_sql(
        "sales",
        engine,
        if_exists="append",
        index=False,
        method="multi",
        chunksize=1000,
    )
    print(f"   ✅ {len(sales_out):,} فاتورة")

    # ---- 2) خريطة invoice_no → sale_id ----
    print("⏳ ربط sale_id...")
    with engine.connect() as conn:
        mapping = pd.read_sql(
            text("SELECT sale_id, invoice_no FROM sales WHERE invoice_no IS NOT NULL"),
            conn,
        )
    inv_to_sale = dict(zip(mapping["invoice_no"].astype(str), mapping["sale_id"]))

    # ---- 3) sale_items ----
    print("⏳ إدخال sale_items...")
    items = df.copy()
    items["sale_id"] = items["InvoiceNo"].map(inv_to_sale)
    items = items.dropna(subset=["sale_id"])
    items["sale_id"] = items["sale_id"].astype(int)
    items["discount"] = 0

    items_out = items[[
        "sale_id", "product_id", "Quantity", "UnitPrice", "discount"
    ]].rename(columns={
        "Quantity": "quantity",
        "UnitPrice": "unit_price",
    })

    items_out.to_sql(
        "sale_items",
        engine,
        if_exists="append",
        index=False,
        method="multi",
        chunksize=2000,
    )
    print(f"   ✅ {len(items_out):,} عنصر")

    # ---- تحقق ----
    print("\n" + "=" * 45)
    print("📊 النتيجة النهائية:")
    print(f"   sales      → {count(engine, 'sales'):,}")
    print(f"   sale_items → {count(engine, 'sale_items'):,}")
    print("=" * 45)
    print("🎉 تم الإدخال بنجاح!")


if __name__ == "__main__":
    run()