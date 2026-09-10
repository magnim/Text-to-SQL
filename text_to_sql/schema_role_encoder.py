class SchemaRoleEncoder:
    NORMAL = 0
    TABLE = 1
    COLUMN = 2
    QUESTION = 3
    SQL = 4
    NUMBER_OF_ROLES = 5

    def __init__(self, tokenizer):
        if tokenizer is None:
            raise ValueError("tokenizer cannot be None.")
        self.tokenizer = tokenizer

    def _add(self, text, role, ids, roles):
        token_ids = self.tokenizer.encode_ids(text)
        ids.extend(token_ids)
        roles.extend([role] * len(token_ids))

    def encode_schema_with_spans(self, schema, include_header=True):
        if not isinstance(schema, dict) or not schema:
            raise ValueError("schema must be a non-empty dictionary.")
        ids, roles = [], []
        spans = {"tables": {}, "columns": {}}
        if include_header:
            self._add("Schema:\n", self.NORMAL, ids, roles)
        for table, columns in schema.items():
            self._add("Table ", self.NORMAL, ids, roles)
            start = len(ids)
            self._add(table, self.TABLE, ids, roles)
            spans["tables"][table] = (start, len(ids))
            self._add(": ", self.NORMAL, ids, roles)
            for index, column in enumerate(columns):
                start = len(ids)
                self._add(column, self.COLUMN, ids, roles)
                spans["columns"][(table, column)] = (start, len(ids))
                if index < len(columns) - 1:
                    self._add(", ", self.NORMAL, ids, roles)
            self._add(".\n", self.NORMAL, ids, roles)
        return ids, roles, spans

    def encode_schema(self, schema):
        ids, roles, _ = self.encode_schema_with_spans(schema, include_header=False)
        return ids, roles

    def encode_prompt_with_spans(self, schema, question):
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question cannot be empty.")
        ids, roles, spans = self.encode_schema_with_spans(schema, include_header=True)
        self._add("\nQuestion:\n", self.NORMAL, ids, roles)
        q_start = len(ids)
        self._add(question.strip(), self.QUESTION, ids, roles)
        spans["question"] = (q_start, len(ids))
        self._add("\n\nSQL:\n", self.NORMAL, ids, roles)
        return ids, roles, spans

    def encode_prompt(self, schema, question):
        ids, roles, _ = self.encode_prompt_with_spans(schema, question)
        return ids, roles

    def append_sql_prefix(self, ids, roles, sql):
        sql_ids = self.tokenizer.encode_ids(sql)
        return ids + sql_ids, roles + [self.SQL] * len(sql_ids)
