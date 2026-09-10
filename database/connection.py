import os
import sqlite3
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*args, **kwargs):
        return False


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


def _resolve_sqlite_path() -> Path:
    raw_path = os.getenv("DB_PATH", "text_to_sql.db").strip()
    db_path = Path(raw_path).expanduser()
    if not db_path.is_absolute():
        db_path = PROJECT_ROOT / db_path
    return db_path.resolve()


def get_connection():
    engine = os.getenv("DB_ENGINE", "mysql").strip().lower()

    if engine == "sqlite":
        db_path = _resolve_sqlite_path()
        connection = sqlite3.connect(str(db_path), check_same_thread=False)
        connection.row_factory = sqlite3.Row
        return connection

    if engine != "mysql":
        raise ValueError(f"Unsupported DB_ENGINE: {engine}")

    import pymysql

    return pymysql.connect(
        host=os.getenv("DB_HOST"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        database=os.getenv("DB_NAME"),
        cursorclass=pymysql.cursors.DictCursor,
    )
