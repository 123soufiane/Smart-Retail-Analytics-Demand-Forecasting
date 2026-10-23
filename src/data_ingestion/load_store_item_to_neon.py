"""Load the Store Item Demand Forecasting data into Neon.

The loader is designed for the 1 GB Neon project limit and for slow/interrupted
network sessions:

* Each source store is loaded atomically in its own transaction. A store is
  therefore either fully present or absent -- never half-loaded.
* Rows are written with PostgreSQL ``COPY`` into a staging table and then moved
  into ``sales`` / ``sale_items`` with set-based SQL, which is much faster than
  pandas ``to_sql`` with many small batches.
* The loader checks the database size before each store and stops cleanly when
  the next store would not fit (``STORE_ITEM_SIZE_LIMIT_MB``). Stores that were
  loaded stay loaded, so rerunning the command later continues from there.
* Stores already present with the expected number of rows are skipped, so
  rerunning never duplicates data.
* Sale ids are assigned explicitly, so the large ``sales.invoice_no`` index is
  not required and ~240 MB of database space is saved.
"""

import io
import os
from collections import Counter
from pathlib import Path

import pandas as pd
from sqlalchemy import bindparam, text

from src.database.connection import get_engine

DATA_PATH = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "raw"
    / "store_item"
    / "retail_sales.csv"
)
SOURCE_PREFIX = "SITEM-"
READ_CHUNKSIZE = 200_000
STAGE_TABLE = "store_item_stage"
STAGE_COLUMNS = (
    "sale_id, invoice_no, store_id, sale_date, total_amount, "
    "product_id, quantity, unit_price"
)
# Stop loading a little below Neon's 1 GB project limit, leaving room for WAL.
SIZE_LIMIT_MB = float(os.getenv("STORE_ITEM_SIZE_LIMIT_MB", "990"))
STORE_SIZE_ESTIMATE_MB = 30.0


def count(engine, table):
    with engine.connect() as conn:
        return conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()


def db_size_mb(engine):
    with engine.connect() as conn:
        size = conn.execute(
            text("SELECT pg_database_size(current_database())")
        ).scalar()
    return size / 1024 / 1024


def scan_source(path):
    """Validate the CSV once and collect catalog metadata.

    Returns:
        store_counts: number of source rows per store_id
        store_first_dates: first sale date per store_id
        item_average_prices: mean price per item_id
    """
    store_counts = Counter()
    store_first_dates = {}
    price_sums = {}
    price_counts = {}

    for frame in pd.read_csv(path, chunksize=READ_CHUNKSIZE, usecols=[
        "date", "store_id", "item_id", "sales", "price"
    ]):
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        frame["sales"] = pd.to_numeric(frame["sales"], errors="coerce")
        frame["price"] = pd.to_numeric(frame["price"], errors="coerce")
        if frame[["date", "store_id", "item_id", "sales", "price"]].isna().any().any():
            raise ValueError("يوجد تاريخ أو عدد مبيعات أو سعر غير صالح في ملف البيانات")
        if (frame["sales"] < 0).any() or (frame["sales"] % 1 != 0).any():
            raise ValueError("يجب أن تكون المبيعات أعداداً صحيحة غير سالبة")
        if (frame["price"] < 0).any():
            raise ValueError("يجب ألا تكون الأسعار سالبة")

        frame["store_id"] = frame["store_id"].astype(str)
        frame["item_id"] = frame["item_id"].astype(str)
        store_counts.update(frame["store_id"].value_counts().to_dict())

        for store_id, first_date in frame.groupby("store_id")["date"].min().items():
            previous = store_first_dates.get(store_id)
            if previous is None or first_date < previous:
                store_first_dates[store_id] = first_date

        stats = frame.groupby("item_id")["price"].agg(["sum", "count"])
        for item_id, row in stats.iterrows():
            price_sums[item_id] = price_sums.get(item_id, 0.0) + float(row["sum"])
            price_counts[item_id] = price_counts.get(item_id, 0) + int(row["count"])

    if not store_counts:
        raise ValueError(f"ملف البيانات فارغ: {path}")

    average_prices = {
        item_id: round(price_sums[item_id] / price_counts[item_id], 2)
        for item_id in price_sums
    }
    return dict(store_counts), store_first_dates, average_prices


def ensure_references(engine):
    with engine.begin() as conn:
        category_id = conn.execute(
            text("SELECT category_id FROM categories WHERE category_name = :name"),
            {"name": "Store Item"},
        ).scalar()
        if category_id is None:
            category_id = conn.execute(
                text(
                    "INSERT INTO categories (category_name) VALUES (:name) "
                    "RETURNING category_id"
                ),
                {"name": "Store Item"},
            ).scalar()

        supplier_id = conn.execute(
            text(
                "SELECT supplier_id FROM suppliers WHERE supplier_name = :name "
                "ORDER BY supplier_id LIMIT 1"
            ),
            {"name": "Store Item Dataset"},
        ).scalar()
        if supplier_id is None:
            supplier_id = conn.execute(
                text(
                    "INSERT INTO suppliers (supplier_name, country, email) "
                    "VALUES (:name, 'Unknown', 'store-item@dataset.local') "
                    "RETURNING supplier_id"
                ),
                {"name": "Store Item Dataset"},
            ).scalar()

    return category_id, supplier_id


def ensure_stores(engine, store_first_dates):
    names = {store_id: f"Store Item {store_id}" for store_id in store_first_dates}
    with engine.begin() as conn:
        existing = dict(
            conn.execute(
                text(
                    "SELECT store_name, store_id FROM stores "
                    "WHERE store_name IN :names"
                ).bindparams(bindparam("names", expanding=True)),
                {"names": list(names.values())},
            ).fetchall()
        )
        for source_id, name in names.items():
            if name in existing:
                continue
            existing[name] = conn.execute(
                text(
                    "INSERT INTO stores (store_name, city, country, opened_at) "
                    "VALUES (:name, 'Unknown', 'Unknown', :opened_at) "
                    "RETURNING store_id"
                ),
                {
                    "name": name,
                    "opened_at": store_first_dates[source_id].date(),
                },
            ).scalar()

    return {source_id: existing[name] for source_id, name in names.items()}


def ensure_products(engine, average_prices, category_id, supplier_id):
    skus = {item_id: f"SITEM-{item_id}" for item_id in average_prices}
    with engine.begin() as conn:
        existing = dict(
            conn.execute(
                text("SELECT sku, product_id FROM products WHERE sku IN :skus")
                .bindparams(bindparam("skus", expanding=True)),
                {"skus": list(skus.values())},
            ).fetchall()
        )
        for item_id, sku in skus.items():
            if sku in existing:
                continue
            existing[sku] = conn.execute(
                text(
                    "INSERT INTO products "
                    "(product_name, category_id, supplier_id, sku, unit_price, cost_price) "
                    "VALUES (:name, :category_id, :supplier_id, :sku, :price, :price) "
                    "RETURNING product_id"
                ),
                {
                    "name": f"Item {item_id}",
                    "category_id": category_id,
                    "supplier_id": supplier_id,
                    "sku": sku,
                    "price": average_prices[item_id],
                },
            ).scalar()

    return {item_id: existing[sku] for item_id, sku in skus.items()}


def prepare_stage(engine):
    """Create the staging table used by COPY."""
    with engine.begin() as conn:
        conn.execute(
            text(
                f"CREATE UNLOGGED TABLE IF NOT EXISTS {STAGE_TABLE} ("
                "sale_id BIGINT, "
                "invoice_no TEXT, "
                "store_id BIGINT, "
                "sale_date DATE, "
                "total_amount NUMERIC(14, 2), "
                "product_id BIGINT, "
                "quantity INTEGER, "
                "unit_price NUMERIC(12, 2))"
            )
        )
        conn.execute(text(f"TRUNCATE {STAGE_TABLE}"))
        conn.execute(
            text(
                "ALTER TABLE sales "
                "ADD COLUMN IF NOT EXISTS invoice_no VARCHAR(100)"
            )
        )


def existing_counts(engine, store_ids):
    store_ids = list(store_ids)
    if not store_ids:
        return {}
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT store_id, COUNT(*) FROM sales "
                "WHERE invoice_no LIKE :pattern AND store_id IN :ids "
                "GROUP BY store_id"
            ).bindparams(bindparam("ids", expanding=True)),
            {"pattern": SOURCE_PREFIX + "%", "ids": store_ids},
        ).fetchall()
    return {store_id: rows_count for store_id, rows_count in rows}


def clear_partial_stores(engine, store_ids):
    """Remove rows of partially loaded stores.

    ``sale_items`` has no index on ``sale_id``, so the ON DELETE CASCADE would
    scan the whole table once per deleted sale. A temporary index makes the
    cleanup fast; it is dropped in the same transaction.
    """
    if not store_ids:
        return
    print(f"♻️  تنظيف {len(store_ids)} متجر غير مكتمل...")
    with engine.begin() as conn:
        conn.execute(
            text(
                "CREATE INDEX IF NOT EXISTS sale_items_sale_id_tmp_idx "
                "ON sale_items (sale_id)"
            )
        )
        for store_id in store_ids:
            conn.execute(
                text(
                    "DELETE FROM sales "
                    "WHERE store_id = :store_id AND invoice_no LIKE :pattern"
                ),
                {"store_id": store_id, "pattern": SOURCE_PREFIX + "%"},
            )
        conn.execute(text("DROP INDEX IF EXISTS sale_items_sale_id_tmp_idx"))


def iter_stores(path, wanted):
    """Yield ``(store_id, rows)`` one complete store at a time."""
    buffer = []
    current = None
    for frame in pd.read_csv(path, chunksize=READ_CHUNKSIZE):
        frame = frame[frame["store_id"].astype(str).isin(wanted)]
        if frame.empty:
            continue
        for store_id, group in frame.groupby("store_id", sort=False):
            store_id = str(store_id)
            if store_id != current:
                if buffer:
                    yield current, pd.concat(buffer, ignore_index=True)
                buffer = []
                current = store_id
            buffer.append(group)
    if buffer:
        yield current, pd.concat(buffer, ignore_index=True)


def load_store(engine, frame, store_map, product_map, next_sale_id):
    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame["store_id"] = frame["store_id"].astype(str)
    frame["item_id"] = frame["item_id"].astype(str)
    frame["sale_id"] = range(next_sale_id, next_sale_id + len(frame))
    frame["store_id_db"] = frame["store_id"].map(store_map)
    frame["product_id"] = frame["item_id"].map(product_map)
    frame["quantity"] = frame["sales"].astype(int)
    frame["unit_price"] = frame["price"].astype(float)
    frame["invoice_no"] = (
        SOURCE_PREFIX
        + frame["store_id"]
        + "-"
        + frame["item_id"]
        + "-"
        + frame["date"].dt.strftime("%Y%m%d")
    )
    frame["total_amount"] = (frame["quantity"] * frame["unit_price"]).round(2)

    stage = frame[
        [
            "sale_id",
            "invoice_no",
            "store_id_db",
            "date",
            "total_amount",
            "product_id",
            "quantity",
            "unit_price",
        ]
    ]
    if stage.isna().any().any():
        raise RuntimeError("تعذر ربط بعض الصفوف بمعرّفات المتاجر أو المنتجات")

    buffer = io.StringIO()
    stage.to_csv(buffer, index=False, header=False, date_format="%Y-%m-%d")
    buffer.seek(0)

    with engine.begin() as conn:
        raw = conn.connection.driver_connection
        with raw.cursor() as cursor:
            copy_sql = (
                f"COPY {STAGE_TABLE} ({STAGE_COLUMNS}) "
                "FROM STDIN WITH (FORMAT csv)"
            )
            if hasattr(cursor, "copy_expert"):
                # psycopg2
                cursor.copy_expert(copy_sql, buffer)
            else:
                # psycopg 3
                with cursor.copy(copy_sql) as copy:
                    copy.write(buffer.getvalue())
            cursor.execute(
                "INSERT INTO sales "
                "(sale_id, customer_id, store_id, sale_date, payment_method, "
                " total_amount, invoice_no) "
                "SELECT sale_id, NULL, store_id, "
                "       sale_date::timestamp AT TIME ZONE 'UTC', "
                f"       'Store Item', total_amount, invoice_no "
                f"FROM {STAGE_TABLE}"
            )
            cursor.execute(
                "INSERT INTO sale_items "
                "(sale_id, product_id, quantity, unit_price, discount) "
                "SELECT sale_id, product_id, quantity, unit_price, 0 "
                f"FROM {STAGE_TABLE} WHERE quantity > 0"
            )
            cursor.execute(f"TRUNCATE {STAGE_TABLE}")

    return next_sale_id + len(frame)


def sync_sale_id_sequence(engine):
    with engine.begin() as conn:
        sequence = conn.execute(
            text("SELECT pg_get_serial_sequence('sales', 'sale_id')")
        ).scalar()
        if sequence:
            conn.execute(
                text(f"SELECT setval('{sequence}', "
                     "GREATEST((SELECT COALESCE(MAX(sale_id), 1) FROM sales), 1))")
            )


def run():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"غير موجود: {DATA_PATH}")

    print("⏳ فحص ملف المصدر...")
    store_counts, store_first_dates, average_prices = scan_source(DATA_PATH)
    total_rows = sum(store_counts.values())
    print(
        f"   بيانات المصدر: {total_rows:,} صف، "
        f"{len(store_counts)} متجر، {len(average_prices)} منتج"
    )

    engine = get_engine()
    category_id, supplier_id = ensure_references(engine)
    store_map = ensure_stores(engine, store_first_dates)
    product_map = ensure_products(engine, average_prices, category_id, supplier_id)
    prepare_stage(engine)

    print("🔎 فحص ما هو محمّل مسبقاً في Neon...")
    loaded = existing_counts(engine, store_map.values())

    complete = []
    partial = []
    pending = []
    for source_id, expected in store_counts.items():
        db_id = store_map[source_id]
        current = loaded.get(db_id, 0)
        if current == expected:
            complete.append(source_id)
        elif current:
            partial.append((source_id, db_id, current, expected))
            pending.append(source_id)
        else:
            pending.append(source_id)

    for source_id, _db_id, current, expected in partial:
        print(
            f"♻️  {source_id}: {current:,}/{expected:,} صف "
            "→ سيتم إعادة تحميله بالكامل"
        )
    clear_partial_stores(engine, [db_id for _, db_id, _, _ in partial])

    if not pending:
        print("✅ جميع بيانات Store Item محمّلة مسبقاً — لا شيء للتحميل")
        return

    print(
        f"⏳ متاجر مكتملة: {len(complete)}، متاجر للتحميل: {len(pending)}"
    )
    size_mb = db_size_mb(engine)
    print(f"💾 حجم قاعدة البيانات: {size_mb:.0f} MB / حد التحميل {SIZE_LIMIT_MB:.0f} MB")

    next_sale_id = 1
    with engine.connect() as conn:
        next_sale_id = (
            conn.execute(text("SELECT COALESCE(MAX(sale_id), 0) FROM sales")).scalar()
            + 1
        )

    estimate = STORE_SIZE_ESTIMATE_MB
    loaded_stores = []
    skipped_stores = []
    for source_id, frame in iter_stores(DATA_PATH, set(pending)):
        size_mb = db_size_mb(engine)
        if size_mb + estimate > SIZE_LIMIT_MB:
            skipped_stores.append(source_id)
            print(
                f"⛔ لا توجد مساحة كافية لـ {source_id} "
                f"({size_mb:.0f} MB + ~{estimate:.0f} MB > {SIZE_LIMIT_MB:.0f} MB)"
            )
            continue

        before = size_mb
        next_sale_id = load_store(
            engine, frame, store_map, product_map, next_sale_id
        )
        after = db_size_mb(engine)
        estimate = max(after - before, 1.0)
        loaded_stores.append(source_id)
        print(
            f"   ✅ {source_id}: {len(frame):,} صف | "
            f"{after:.0f} MB (+{after - before:.0f} MB)"
        )

    sync_sale_id_sequence(engine)

    with engine.begin() as conn:
        conn.execute(text(f"DROP TABLE IF EXISTS {STAGE_TABLE}"))

    print("\n📊 الحالة النهائية:")
    for table in ("stores", "products", "sales", "sale_items"):
        print(f"   {table:12} → {count(engine, table):,}")

    loaded_now = existing_counts(engine, store_map.values())
    still_missing = [
        source_id
        for source_id, expected in store_counts.items()
        if loaded_now.get(store_map[source_id], 0) != expected
    ]
    if still_missing:
        print(
            f"\n⚠️  لم تُحمَّل {len(still_missing)} متجر بسبب حد التخزين: "
            f"{', '.join(still_missing[:10])}"
            f"{' ...' if len(still_missing) > 10 else ''}"
        )
        print(
            "   حرّر مساحة (مثلاً احذف متجراً لا تحتاجه) ثم أعد تشغيل نفس الأمر "
            "لمتابعة التحميل من حيث توقف."
        )
    else:
        print("✅ تم تحميل جميع بيانات Store Item إلى Neon")

    return {"loaded": loaded_stores, "missing": still_missing}


if __name__ == "__main__":
    run()
