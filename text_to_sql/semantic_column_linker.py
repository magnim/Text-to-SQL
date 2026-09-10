import re

import torch


class SemanticColumnLinker:
    def __init__(self, model, tokenizer, lexical_weight: float = 3.0):
        self.model = model
        self.tokenizer = tokenizer
        self.lexical_weight = lexical_weight

    def _encode(self, text: str, max_length: int):
        pad_id = self.tokenizer.vocab["<PAD>"]

        token_ids = self.tokenizer.encode_ids(text)[:max_length]
        mask = [1] * len(token_ids)

        padding = max_length - len(token_ids)

        token_ids += [pad_id] * padding
        mask += [0] * padding

        return (
            torch.tensor([token_ids], dtype=torch.long),
            torch.tensor([mask], dtype=torch.long),
        )

    def _extract_context(self, question: str, value, window: int = 5) -> str:
        if value is None:
            return question

        tokens = re.findall(r"[A-Za-z_]+|\d+(?:\.\d+)?|[^\w\s]", question)

        for index, token in enumerate(tokens):
            if not re.fullmatch(r"\d+(?:\.\d+)?", token):
                continue

            if float(token) == float(value):
                start = max(0, index - window)
                end = min(len(tokens), index + window + 1)
                return " ".join(tokens[start:end])

        return question

    def _lexical_score(self, context: str, column: str) -> float:
        context_tokens = set(re.findall(r"[A-Za-z0-9]+", context.lower()))
        column_tokens = re.findall(r"[A-Za-z0-9]+", column.lower().replace("_", " "))

        if not column_tokens:
            return 0.0

        matches = sum(token in context_tokens for token in column_tokens)
        return matches / len(column_tokens)

    def rank_columns(
        self,
        question: str,
        table: str,
        columns: list[str],
        value=None,
    ) -> list[tuple[str, float]]:
        context = self._extract_context(question, value)
        ranked = []

        self.model.eval()

        with torch.no_grad():
            for column in columns:
                question_ids, question_mask = self._encode(context, 64)

                column_ids, column_mask = self._encode(
                    f"{table} {column}",
                    16,
                )

                neural_score = self.model(
                    question_ids=question_ids,
                    question_mask=question_mask,
                    column_ids=column_ids,
                    column_mask=column_mask,
                ).item()

                lexical_score = self._lexical_score(context, column)

                final_score = (
                    neural_score
                    + self.lexical_weight * lexical_score
                )

                ranked.append((column, final_score))

        ranked.sort(key=lambda item: item[1], reverse=True)

        return ranked

# TESTING
if __name__ == "__main__":
    from pathlib import Path

    from src.tokenizer.bpe import BPETrainer
    from text_to_sql.semantic_column_ranker import SemanticColumnRanker

    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    tokenizer = BPETrainer.load(PROJECT_ROOT / "tokenizer_balanced.json")

    model = SemanticColumnRanker(
        vocabulary_size=len(tokenizer.vocab),
        embedding_dimension=64,
        hidden_dimension=64,
    )

    model.load_state_dict(
        torch.load(
            PROJECT_ROOT / "semantic_column_ranker.pt",
            map_location="cpu",
        )
    )

    model.eval()

    linker = SemanticColumnLinker(
        model=model,
        tokenizer=tokenizer,
        lexical_weight=3.0,
    )

    tests = [
        {
            "question": "show customers older than 30",
            "table": "customers",
            "columns": ["id", "name", "city", "age", "membership_level"],
            "value": 30,
            "expected": "age",
        },
        {
            "question": "show products priced above 50",
            "table": "products",
            "columns": ["id", "name", "category", "price", "stock", "rating"],
            "value": 50,
            "expected": "price",
        },
        {
            "question": "show employees with salary greater than 60000",
            "table": "employees",
            "columns": ["id", "name", "department", "salary", "age", "years_experience"],
            "value": 60000,
            "expected": "salary",
        },
    ]

    for test in tests:
        ranked = linker.rank_columns(
            question=test["question"],
            table=test["table"],
            columns=test["columns"],
            value=test["value"],
        )

        predicted = ranked[0][0]

        print("\nQuestion :", test["question"])
        print("Expected :", test["expected"])
        print("Predicted:", predicted)
        print("Ranking  :", ranked)

        assert predicted == test["expected"]

    print("\nSemanticColumnLinker standalone tests passed.")