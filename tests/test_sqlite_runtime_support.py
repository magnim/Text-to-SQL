
import sqlite3

from database.connection import get_connection
from database.schema_scanner import scan_column_types,scan_schema


def test_sqlite_backend_connection_and_schema_scan(tmp_path, monkeypatch):
    db_path = tmp_path / "runtime.db"
    seed = sqlite3.connect(db_path)
    seed.execute("CREATE TABLE employees (id INTEGER, name TEXT, age INTEGER)")
    seed.execute("INSERT INTO employees VALUES (1, 'Alice', 30)")
    seed.commit()
    seed.close()

    monkeypatch.setenv("DB_ENGINE", "sqlite")
    monkeypatch.setenv("DB_PATH", str(db_path))

    connection = get_connection()
    try:
        schema = scan_schema(connection)
        assert schema == {"employees": ["id", "name", "age"]}
        assert scan_column_types(connection) == {
            ("employees","id"):"INTEGER",
            ("employees","name"):"TEXT",
            ("employees","age"):"INTEGER",
        }

        cursor = connection.cursor()
        cursor.execute("SELECT id, name, age FROM employees")
        row = cursor.fetchone()
        assert dict(row) == {"id": 1, "name": "Alice", "age": 30}
    finally:
        connection.close()



def test_sqlite_connection_supports_streamlit_rerun_thread(tmp_path, monkeypatch):
    import threading

    db_path = tmp_path / "streamlit.db"
    seed = sqlite3.connect(db_path)
    seed.execute("CREATE TABLE employees (id INTEGER, name TEXT)")
    seed.execute("INSERT INTO employees VALUES (1, 'Alice')")
    seed.commit()
    seed.close()

    monkeypatch.setenv("DB_ENGINE", "sqlite")
    monkeypatch.setenv("DB_PATH", str(db_path))

    connection = get_connection()
    errors = []
    rows = []

    def use_connection():
        try:
            cursor = connection.cursor()
            cursor.execute("SELECT name FROM employees")
            rows.extend(cursor.fetchall())
            cursor.close()
        except Exception as error:
            errors.append(error)

    thread = threading.Thread(target=use_connection)
    thread.start()
    thread.join()

    connection.close()
    assert not errors
    assert [row["name"] for row in rows] == ["Alice"]
