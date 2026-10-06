"""Create the database schema from sql/schema.sql."""

from pathlib import Path

from src.database.connection import get_engine


def create_schema():
    schema_path = Path(__file__).resolve().parents[2] / "sql" / "schema.sql"

    if not schema_path.exists():
        raise FileNotFoundError(f"❌ ملف Schema غير موجود: {schema_path}")

    sql = schema_path.read_text(encoding="utf-8")
    engine = get_engine()

    try:
        with engine.begin() as connection:
            connection.exec_driver_sql(sql)
        print("✅ Database schema created successfully!")
    except Exception as e:
        print(f"❌ حدث خطأ أثناء إنشاء الـ Schema:\n{e}")
        raise


if __name__ == "__main__":
    create_schema()