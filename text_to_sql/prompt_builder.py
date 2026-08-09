class PromptBuilder:

    def build_inference_prompt(self,schema_text: str,question: str) -> str:
        if not isinstance(schema_text, str):
            raise TypeError("schema_text must be a string.")
        if not schema_text.strip():
            raise ValueError("schema_text cannot be empty.")
        if not isinstance(question, str):
            raise TypeError("question must be a string.")
        if not question.strip():
            raise ValueError("question cannot be empty.")
        prompt = (
            f"Schema:\n"
            f"{schema_text}\n\n"
            f"Question:\n"
            f"{question.strip()}\n\n"
            f"SQL:\n"
        )
        return prompt

    def build_training_prompt(self,schema_text: str,question: str,sql: str) -> str:

        if not isinstance(schema_text, str):
            raise TypeError("schema_text must be a string.")
        if not schema_text.strip():
            raise ValueError("schema_text cannot be empty.")
        if not isinstance(question, str):
            raise TypeError("question must be a string.")
        if not question.strip():
            raise ValueError("question cannot be empty.")
        if not isinstance(sql, str):
            raise TypeError("sql must be a string.")
        if not sql.strip():
            raise ValueError("sql cannot be empty.")
        prompt = (
            f"Schema:\n"
            f"{schema_text.strip()}\n\n"
            f"Question:\n"
            f"{question.strip()}\n\n"
            f"SQL:\n"
            f"{sql.strip()}"
        )

        return prompt