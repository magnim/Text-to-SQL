import sqlite3

from database.connection import get_connection


def _scan_sqlite_schema(connection: sqlite3.Connection) -> dict[str, list[str]]:
    cursor = connection.cursor()
    try:
        cursor.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
              AND name NOT LIKE 'sqlite_%'
            ORDER BY name
            """
        )
        table_names = [row[0] for row in cursor.fetchall()]

        schema = {}
        for table_name in table_names:
            escaped_table_name = table_name.replace('"', '""')
            cursor.execute(f'PRAGMA table_info("{escaped_table_name}")')
            schema[table_name] = [row[1] for row in cursor.fetchall()]
        return schema
    finally:
        cursor.close()


def _scan_sqlite_column_types(connection: sqlite3.Connection) -> dict[tuple[str, str], str]:
    cursor = connection.cursor()
    try:
        cursor.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
              AND name NOT LIKE 'sqlite_%'
            ORDER BY name
            """
        )
        table_names = [row[0] for row in cursor.fetchall()]

        column_types = {}
        for table_name in table_names:
            escaped_table_name = table_name.replace('"', '""')
            cursor.execute(f'PRAGMA table_info("{escaped_table_name}")')
            for row in cursor.fetchall():
                column_types[(table_name, row[1])] = row[2] or ""
        return column_types
    finally:
        cursor.close()


def _scan_mysql_schema(connection) -> dict[str, list[str]]:
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT TABLE_NAME, COLUMN_NAME
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_SCHEMA = DATABASE()
            ORDER BY TABLE_NAME, ORDINAL_POSITION
            """
        )
        rows = cursor.fetchall()

    schema = {}
    for row in rows:
        table_name = row["TABLE_NAME"]
        column_name = row["COLUMN_NAME"]
        schema.setdefault(table_name, []).append(column_name)
    return schema


def _scan_mysql_column_types(connection) -> dict[tuple[str, str], str]:
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT TABLE_NAME, COLUMN_NAME, DATA_TYPE
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_SCHEMA = DATABASE()
            ORDER BY TABLE_NAME, ORDINAL_POSITION
            """
        )
        rows = cursor.fetchall()

    return {
        (row["TABLE_NAME"], row["COLUMN_NAME"]): row["DATA_TYPE"] or ""
        for row in rows
    }


def scan_schema(connection=None) -> dict[str, list[str]]:
    owns_connection = connection is None
    if connection is None:
        connection = get_connection()

    try:
        if isinstance(connection, sqlite3.Connection):
            return _scan_sqlite_schema(connection)
        return _scan_mysql_schema(connection)
    finally:
        if owns_connection:
            connection.close()


def scan_column_types(connection=None) -> dict[tuple[str, str], str]:
    owns_connection = connection is None
    if connection is None:
        connection = get_connection()

    try:
        if isinstance(connection, sqlite3.Connection):
            return _scan_sqlite_column_types(connection)
        return _scan_mysql_column_types(connection)
    finally:
        if owns_connection:
            connection.close()


if __name__ == "__main__":
    scan_schema()
