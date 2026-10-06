import random
from datetime import datetime, timedelta
from faker import Faker
from sqlalchemy import text
from src.database.connection import get_engine
from src.data_generation.config import CONFIG

fake = Faker()
engine = get_engine()

def clear_tables():
    """مسح البيانات القديمة (اختياري)"""
    tables = [
        "returns", "sale_items", "sales", "purchase_items",
        "suppliers_orders", "inventory", "products", "customers",
        "stores", "suppliers", "categories"
    ]
    with engine.begin() as conn:
        for table in tables:
            conn.execute(text(f"TRUNCATE TABLE {table} RESTART IDENTITY CASCADE;"))
    print("🧹 تم مسح البيانات القديمة")

def generate_categories():
    categories = [
        "Electronics", "Clothing", "Home & Kitchen", "Beauty", "Sports",
        "Toys", "Books", "Grocery", "Furniture", "Automotive",
        "Health", "Jewelry", "Shoes", "Garden", "Pet Supplies"
    ]
    with engine.begin() as conn:
        for name in categories[:CONFIG["num_categories"]]:
            conn.execute(
                text("INSERT INTO categories (category_name) VALUES (:name)"),
                {"name": name}
            )
    print(f"✅ تم إنشاء {CONFIG['num_categories']} تصنيف")

def generate_suppliers():
    with engine.begin() as conn:
        for _ in range(CONFIG["num_suppliers"]):
            conn.execute(
                text("""
                    INSERT INTO suppliers (supplier_name, country, email)
                    VALUES (:name, :country, :email)
                """),
                {
                    "name": fake.company(),
                    "country": fake.country(),
                    "email": fake.company_email()
                }
            )
    print(f"✅ تم إنشاء {CONFIG['num_suppliers']} مورد")

def generate_products():
    with engine.begin() as conn:
        # جلب IDs
        categories = conn.execute(text("SELECT category_id FROM categories")).fetchall()
        suppliers = conn.execute(text("SELECT supplier_id FROM suppliers")).fetchall()

        category_ids = [c[0] for c in categories]
        supplier_ids = [s[0] for s in suppliers]

        for i in range(CONFIG["num_products"]):
            cost = round(random.uniform(5, 300), 2)
            price = round(cost * random.uniform(1.3, 2.5), 2)

            conn.execute(
                text("""
                    INSERT INTO products 
                    (product_name, category_id, supplier_id, sku, unit_price, cost_price)
                    VALUES (:name, :cat, :sup, :sku, :price, :cost)
                """),
                {
                    "name": fake.catch_phrase(),
                    "cat": random.choice(category_ids),
                    "sup": random.choice(supplier_ids),
                    "sku": f"SKU-{10000 + i}",
                    "price": price,
                    "cost": cost
                }
            )
    print(f"✅ تم إنشاء {CONFIG['num_products']} منتج")

def generate_stores():
    with engine.begin() as conn:
        for _ in range(CONFIG["num_stores"]):
            conn.execute(
                text("""
                    INSERT INTO stores (store_name, city, country, opened_at)
                    VALUES (:name, :city, :country, :opened)
                """),
                {
                    "name": f"{fake.company()} Store",
                    "city": fake.city(),
                    "country": fake.country(),
                    "opened": fake.date_between(start_date="-8y", end_date="-1y")
                }
            )
    print(f"✅ تم إنشاء {CONFIG['num_stores']} متجر")

def generate_customers():
    with engine.begin() as conn:
        for _ in range(CONFIG["num_customers"]):
            conn.execute(
                text("""
                    INSERT INTO customers (first_name, last_name, email, city, country)
                    VALUES (:first, :last, :email, :city, :country)
                """),
                {
                    "first": fake.first_name(),
                    "last": fake.last_name(),
                    "email": fake.unique.email(),
                    "city": fake.city(),
                    "country": fake.country()
                }
            )
    print(f"✅ تم إنشاء {CONFIG['num_customers']} عميل")

def generate_inventory():
    with engine.begin() as conn:
        stores = [s[0] for s in conn.execute(text("SELECT store_id FROM stores")).fetchall()]
        products = [p[0] for p in conn.execute(text("SELECT product_id FROM products")).fetchall()]

        for store_id in stores:
            # كل متجر يحتوي على معظم المنتجات
            selected_products = random.sample(products, k=int(len(products) * 0.8))
            for product_id in selected_products:
                conn.execute(
                    text("""
                        INSERT INTO inventory (store_id, product_id, quantity, reorder_level)
                        VALUES (:store, :product, :qty, :reorder)
                    """),
                    {
                        "store": store_id,
                        "product": product_id,
                        "qty": random.randint(5, 200),
                        "reorder": random.randint(5, 20)
                    }
                )
    print("✅ تم إنشاء بيانات المخزون")

def generate_sales_and_items():
    with engine.begin() as conn:
        customers = [c[0] for c in conn.execute(text("SELECT customer_id FROM customers")).fetchall()]
        stores = [s[0] for s in conn.execute(text("SELECT store_id FROM stores")).fetchall()]
        products = conn.execute(text("SELECT product_id, unit_price FROM products")).fetchall()
        product_dict = {p[0]: p[1] for p in products}

        payment_methods = ["Credit Card", "Cash", "Debit Card", "Mobile Payment"]

        for _ in range(CONFIG["num_sales"]):
            sale_date = fake.date_time_between(start_date="-2y", end_date="now")
            customer_id = random.choice(customers)
            store_id = random.choice(stores)
            payment = random.choice(payment_methods)

            # إنشاء الفاتورة
            result = conn.execute(
                text("""
                    INSERT INTO sales (customer_id, store_id, sale_date, payment_method)
                    VALUES (:cust, :store, :date, :pay)
                    RETURNING sale_id
                """),
                {
                    "cust": customer_id,
                    "store": store_id,
                    "date": sale_date,
                    "pay": payment
                }
            )
            sale_id = result.scalar()

            # إضافة منتجات للفاتورة
            num_items = random.randint(1, CONFIG["max_items_per_sale"])
            selected_products = random.sample(list(product_dict.keys()), k=num_items)
            total = 0

            for product_id in selected_products:
                qty = random.randint(1, 4)
                unit_price = float(product_dict[product_id])
                discount = round(random.uniform(0, 15), 2) if random.random() < 0.3 else 0

                conn.execute(
                    text("""
                        INSERT INTO sale_items 
                        (sale_id, product_id, quantity, unit_price, discount)
                        VALUES (:sale, :prod, :qty, :price, :disc)
                    """),
                    {
                        "sale": sale_id,
                        "prod": product_id,
                        "qty": qty,
                        "price": unit_price,
                        "disc": discount
                    }
                )
                total += (unit_price * qty) * (1 - discount/100)

            # تحديث إجمالي الفاتورة
            conn.execute(
                text("UPDATE sales SET total_amount = :total WHERE sale_id = :id"),
                {"total": round(total, 2), "id": sale_id}
            )

    print(f"✅ تم إنشاء {CONFIG['num_sales']} عملية بيع مع تفاصيلها")

def generate_returns():
    with engine.begin() as conn:
        # جلب بعض عمليات البيع مع منتجاتها
        sales = conn.execute(text("""
            SELECT s.sale_id, si.product_id, si.quantity, s.sale_date
            FROM sales s
            JOIN sale_items si ON s.sale_id = si.sale_id
            ORDER BY RANDOM()
            LIMIT :limit
        """), {"limit": CONFIG["num_returns"] * 2}).fetchall()

        reasons = [
            "Defective product", "Wrong size", "Changed mind",
            "Not as described", "Damaged during shipping", "Other"
        ]

        count = 0
        for sale_id, product_id, max_qty, sale_date in sales:
            if count >= CONFIG["num_returns"]:
                break

            return_date = sale_date + timedelta(days=random.randint(1, 30))
            if return_date > datetime.now():
                continue

            qty = random.randint(1, max_qty)

            conn.execute(
                text("""
                    INSERT INTO returns (sale_id, product_id, quantity, return_date, reason)
                    VALUES (:sale, :prod, :qty, :date, :reason)
                """),
                {
                    "sale": sale_id,
                    "prod": product_id,
                    "qty": qty,
                    "date": return_date,
                    "reason": random.choice(reasons)
                }
            )
            count += 1

    print(f"✅ تم إنشاء {count} عملية إرجاع")

def main():
    print("\n🚀 بدء توليد البيانات الاصطناعية...\n")

    # clear_tables()  # أزل التعليق إذا أردت مسح البيانات القديمة

    generate_categories()
    generate_suppliers()
    generate_products()
    generate_stores()
    generate_customers()
    generate_inventory()
    generate_sales_and_items()
    generate_returns()

    print("\n🎉 تم توليد كل البيانات بنجاح!")

if __name__ == "__main__":
    main()