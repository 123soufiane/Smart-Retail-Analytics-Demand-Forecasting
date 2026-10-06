import os
from pathlib import Path

from sqlalchemy import create_engine, text

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(path=None):
        dotenv_path = Path(path) if path else Path(__file__).with_name(".env")
        if not dotenv_path.exists():
            return False

        for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = [part.strip() for part in raw_line.split("=", 1)]
            os.environ.setdefault(key, value.strip("\"'"))

        return True

load_dotenv(Path(__file__).with_name(".env"))
database_url = os.environ.get("DATABASE_URL")
if not database_url:
    raise RuntimeError("DATABASE_URL is not set. Add it to your environment or .env file.")

engine = create_engine(database_url)

with engine.connect() as conn:
    print(conn.execute(text("SELECT version();")).scalar())