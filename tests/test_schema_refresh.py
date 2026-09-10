
import sqlite3
from pathlib import Path

import database.connection as connection_module
from database.schema_scanner import scan_schema
from text_to_sql_runtime import TextToSQLRuntime


def test_relative_sqlite_db_path_is_resolved_from_project_root(tmp_path, monkeypatch):
    project_root = tmp_path / "project"
    database_dir = project_root / "database"
    database_dir.mkdir(parents=True)
    db_path = database_dir / "app.db"

    seed = sqlite3.connect(db_path)
    seed.execute("CREATE TABLE employees (id INTEGER, name TEXT)")
    seed.commit()
    seed.close()

    unrelated_cwd = tmp_path / "launcher"
    unrelated_cwd.mkdir()

    monkeypatch.setenv("DB_ENGINE", "sqlite")
    monkeypatch.setenv("DB_PATH", "database/app.db")
    monkeypatch.chdir(unrelated_cwd)
    monkeypatch.setattr(connection_module, "PROJECT_ROOT", project_root, raising=False)

    connection = connection_module.get_connection()
    try:
        schema = scan_schema(connection)
        assert schema == {"employees": ["id", "name"]}

        active_file = connection.execute("PRAGMA database_list").fetchone()[2]
        assert Path(active_file).resolve() == db_path.resolve()
    finally:
        connection.close()


def test_refresh_schema_reconnects_and_sees_new_sqlite_table(tmp_path, monkeypatch):
    db_path = tmp_path / "runtime.db"

    seed = sqlite3.connect(db_path)
    seed.execute("CREATE TABLE employees (id INTEGER, name TEXT)")
    seed.commit()
    seed.close()

    monkeypatch.setenv("DB_ENGINE", "sqlite")
    monkeypatch.setenv("DB_PATH", str(db_path))

    runtime = TextToSQLRuntime.__new__(TextToSQLRuntime)
    runtime.connection = connection_module.get_connection()
    runtime.schema = scan_schema(runtime.connection)
    runtime.pipeline = object()

    # Avoid loading/rebuilding the model in this focused runtime-state test.
    runtime._build_pipeline = lambda: object()

    old_connection = runtime.connection

    external = sqlite3.connect(db_path)
    external.execute("CREATE TABLE customers (id INTEGER, name TEXT, age INTEGER)")
    external.commit()
    external.close()

    refreshed = runtime.refresh_schema()

    try:
        assert runtime.connection is not old_connection
        assert refreshed == {
            "customers": ["id", "name", "age"],
            "employees": ["id", "name"],
        }
        assert runtime.schema == refreshed
    finally:
        runtime.connection.close()
