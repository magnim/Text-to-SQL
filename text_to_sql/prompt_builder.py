class PromptBuilder:
    @staticmethod
    def build_inference_prompt(schema, question):
        lines = ["Schema:"]
        for table, columns in schema.items():
            lines.append(f"Table {table}: {', '.join(columns)}.")
        lines.extend(["", "Question:", question.strip(), "", "SQL:", ""])
        return "\n".join(lines)

    @staticmethod
    def build_training_prompt(schema, question, sql):
        return PromptBuilder.build_inference_prompt(schema, question) + sql

    @staticmethod
    def build_question_prompt(question):
        return f"Question:\\n{question.strip()}\\n\\nSQL:\\n"
