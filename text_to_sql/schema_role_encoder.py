from from_scratch.embeddings.schema_role_embedding import SchemaRoleEmbedding

class SchemaRoleEncoder:

    def __init__(self, tokenizer) -> None:
        if tokenizer is None:
            raise ValueError("tokenizer cannot be None.")

        self.tokenizer = tokenizer

    def _encode_segment(self,text: str,role_id: int,token_ids: list[int],role_ids: list[int]) -> None:
        segment_token_ids = (self.tokenizer.encode_ids(text))
        token_ids.extend(segment_token_ids)
        role_ids.extend([role_id] * len(segment_token_ids))

    def encode_schema(self,schema: dict[str, list[str]]) -> tuple[list[int], list[int]]:

        if not isinstance(schema, dict):
            raise TypeError("schema must be a dictionary.")
        if not schema:
            raise ValueError("schema cannot be empty.")

        token_ids = []
        role_ids = []

        for table_name, columns in schema.items():
            self._encode_segment(text="Table ",role_id=SchemaRoleEmbedding.NORMAL,token_ids=token_ids,role_ids=role_ids)
            self._encode_segment(text=table_name,role_id=SchemaRoleEmbedding.TABLE,token_ids=token_ids,role_ids=role_ids)
            self._encode_segment(text=": ",role_id=SchemaRoleEmbedding.NORMAL,token_ids=token_ids,role_ids=role_ids)

            for column_index, column_name in enumerate(columns):
                self._encode_segment(
                    text=column_name,role_id=SchemaRoleEmbedding.COLUMN,token_ids=token_ids,role_ids=role_ids)
                if column_index < len(columns) - 1:
                    self._encode_segment(text=", ",role_id=SchemaRoleEmbedding.NORMAL,token_ids=token_ids,role_ids=role_ids)

            self._encode_segment(text=".\n",role_id=SchemaRoleEmbedding.NORMAL,token_ids=token_ids,role_ids=role_ids)

        if len(token_ids) != len(role_ids):
            raise RuntimeError("Token IDs and role IDs are not aligned.")

        return token_ids, role_ids
