class SchemaEncoder:
    def encode_schema(self,schema: dict[str, list[str]]) -> str:
        if not isinstance(schema, dict):
            raise TypeError("schema must be a dictionary.")

        if not schema:
            raise ValueError("schema cannot be empty.")
        table_descriptions = []
        for table_name, columns in schema.items():
            if not isinstance(table_name, str):
                raise TypeError("Every table name must be a string.")
            if not table_name:
                raise ValueError("Table names cannot be empty.")
            if not isinstance(columns, list):
                raise TypeError(f"Columns for table '{table_name}' must be a list.")
            if not columns:
                raise ValueError(f"Table '{table_name}' must contain at least one column.")
            if any(not isinstance(column, str) for column in columns):
                raise TypeError(f"Every column in table '{table_name}' must be a string.")
            column_text = ", ".join(columns)
            table_description = f"Table {table_name}: {column_text}."
            table_descriptions.append(table_description)
        return "\n".join(table_descriptions)