

"""
تحميل بيانات UCI إلى Neon بشكل ذكي:
- الجداول الممتلئة → تُتخطى
- الجداول الفارغة → تُملأ فقط
"""
import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import text
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from src.database.connection import get_engine


def table_count(engine, table: str) -> int:
    with engine.connect() as conn:
        return conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()


def load_and_clean_uci(path="data/raw/OnlineRetail.xlsx"):
    print("⏳ قراءة وتنظيف بيانات UCI...")
    df = pd.read_excel(path, engine="openpyxl")

    df = df.dropna(subset=["InvoiceNo", "StockCode", "Quantity", "InvoiceDate", "UnitPrice"])
    df["InvoiceNo"] = df["InvoiceNo"].astype(str)
    df = df[~df["InvoiceNo"].str.startswith("C")]
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["StockCode"] = df["StockCode"].astype(str).str.strip()
    df["Description"] = df["Description"].astype(str).str.strip()
    df["CustomerID"] = df["CustomerID"].fillna(-1).astype(int)

    print(f"   الصفوف بعد التنظيف: {len(df):,}")
    return df


def ensure_reference_data(engine, df):
    """إدخال Categories / Suppliers / Stores / Products / Customers فقط إذا كانت فارغة"""

    with engine.begin() as conn:
        # ---- Categories ----
        if table_count(engine, "categories") == 0:
            print("⏳ إدخال categories...")
            conn.execute(text("INSERT INTO categories (category_name) VALUES ('General')"))
        else:
            print("⏭️  categories موجودة → تخطي")

        category_id = conn.execute(text(
            "SELECT category_id FROM categories ORDER BY category_id LIMIT 1"
        )).scalar()

        # ---- Suppliers ----
        if table_count(engine, "suppliers") == 0:
            print("⏳ إدخال suppliers...")
            conn.execute(text("""
                INSERT INTO suppliers (supplier_name, country, email)
                VALUES ('UCI Online Retail', 'United Kingdom', 'uci@example.com')
            """))
        else:
            print("⏭️  suppliers موجودة → تخطي")

        supplier_id = conn.execute(text(
            "SELECT supplier_id FROM suppliers ORDER BY supplier_id LIMIT 1"
        )).scalar()

        # ---- Stores ----
        if table_count(engine, "stores") == 0:
            print("⏳ إدخال stores...")
            conn.execute(text("""
                INSERT INTO stores (store_name, city, country, opened_at)
                VALUES ('UCI Online Store', 'London', 'United Kingdom', '2010-01-01')
            """))
        else:
            print("⏭️  stores موجودة → تخطي")

        store_id = conn.execute(text(
            "SELECT store_id FROM stores ORDER BY store_id LIMIT 1"
        )).scalar()

        # ---- Products ----
        if table_count(engine, "products") == 0:
            products = (
                df.groupby("StockCode")
                .agg(
                    product_name=("Description", lambda x: x.mode().iloc[0] if len(x.mode()) else x.iloc[0]),
                    unit_price=("UnitPrice", "median"),
                )
                .reset_index()
            )
            print(f"⏳ إدخال {len(products):,} منتج...")
            for _, row in products.iterrows():
                cost = round(float(row["unit_price"]) * 0.6, 2)
                conn.execute(
                    text("""
                        INSERT INTO products
                        (product_name, category_id, supplier_id, sku, unit_price, cost_price)
                        VALUES (:name, :cat, :sup, :sku, :price, :cost)
                    """),
                    {
                        "name": str(row["product_name"])[:250],
                        "cat": category_id,
                        "sup": supplier_id,
                        "sku": str(row["StockCode"])[:100],
                        "price": round(float(row["unit_price"]), 2),
                        "cost": cost,
                    },
                )
        else:
            print(f"⏭️  products موجودة ({table_count(engine, 'products'):,}) → تخطي")

        # ---- Customers ----
        if table_count(engine, "customers") == 0:
            customers = (
                df[df["CustomerID"] > 0]
                .groupby("CustomerID", as_index=False)
                .agg(Country=("Country", "first"))
            )
            print(f"⏳ إدخال {len(customers):,} عميل...")
            for _, row in customers.iterrows():
                conn.execute(
                    text("""
                        INSERT INTO customers (first_name, last_name, email, city, country)
                        VALUES (:first, :last, :email, :city, :country)
                        ON CONFLICT (email) DO NOTHING
                    """),
                    {
                        "first": "Customer",
                        "last": str(int(row["CustomerID"])),
                        "email": f"customer_{int(row['CustomerID'])}@uci.local",
                        "city": "Unknown",
                        "country": str(row["Country"])[:100],
                    },
                )
        else:
            print(f"⏭️  customers موجودة ({table_count(engine, 'customers'):,}) → تخطي")

    return store_id


def insert_sales_if_empty(engine, df, store_id):
    """إدخال المبيعات فقط إذا كان جدول sales فاضي"""

    if table_count(engine, "sales") > 0:
        n = table_count(engine, "sales")
        print(f"⏭️  sales موجودة ({n:,}) → تخطي إدخال المبيعات")
        return

    print("⏳ إدخال المبيعات (الجدول فاضي)...")

    with engine.begin() as conn:
        sku_map = {
            r[0]: r[1]
            for r in conn.execute(text("SELECT sku, product_id FROM products")).fetchall()
        }

        cust_map = {}
        for last_name, cid in conn.execute(
            text("SELECT last_name, customer_id FROM customers WHERE first_name = 'Customer'")
        ):
            try:
                cust_map[int(last_name)] = cid
            except Exception:
                pass

        invoices = df.groupby("InvoiceNo")
        total_invoices = len(invoices)
        count = 0

        for invoice_no, group in invoices:
            count += 1
            if count % 2000 == 0:
                print(f"   → {count:,} / {total_invoices:,}")

            first = group.iloc[0]
            cust_id = cust_map.get(int(first["CustomerID"])) if first["CustomerID"] > 0 else None

            result = conn.execute(
                text("""
                    INSERT INTO sales (customer_id, store_id, sale_date, payment_method, total_amount)
                    VALUES (:cust, :store, :date, :pay, 0)
                    RETURNING sale_id
                """),
                {
                    "cust": cust_id,
                    "store": store_id,
                    "date": first["InvoiceDate"],
                    "pay": "Online",
                },
            )
            sale_id = result.scalar()

            total = 0.0
            for _, item in group.iterrows():
                product_id = sku_map.get(str(item["StockCode"]))
                if not product_id:
                    continue

                qty = int(item["Quantity"])
                price = float(item["UnitPrice"])
                total += qty * price

                conn.execute(
                    text("""
                        INSERT INTO sale_items
                        (sale_id, product_id, quantity, unit_price, discount)
                        VALUES (:sale, :prod, :qty, :price, 0)
                    """),
                    {
                        "sale": sale_id,
                        "prod": product_id,
                        "qty": qty,
                        "price": price,
                    },
                )

            conn.execute(
                text("UPDATE sales SET total_amount = :total WHERE sale_id = :id"),
                {"total": round(total, 2), "id": sale_id},
            )

    print(f"✅ تم إدخال {count:,} فاتورة")


def run():
    engine = get_engine()

    print("=" * 55)
    print("🚀 تحميل ذكي لبيانات UCI → Neon")
    print("=" * 55)

    df = load_and_clean_uci()
    store_id = ensure_reference_data(engine, df)
    insert_sales_if_empty(engine, df, store_id)

    print("\n" + "=" * 55)
    print("📊 الحالة النهائية:")
    for t in ["categories", "products", "customers", "stores", "sales", "sale_items"]:
        print(f"   {t:15} → {table_count(engine, t):,}")
    print("=" * 55)
    print("🎉 تم!")


if __name__ == "__main__":
    run()