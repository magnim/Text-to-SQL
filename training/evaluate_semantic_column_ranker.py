import json
from collections import defaultdict
from pathlib import Path
import re

import torch

from src.tokenizer.bpe import BPETrainer
from text_to_sql.semantic_column_ranker import SemanticColumnRanker


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEV_PATH = PROJECT_ROOT / "data" / "column_linking" / "dev.jsonl"
MODEL_PATH = PROJECT_ROOT / "semantic_column_ranker.pt"
TOKENIZER_PATH = PROJECT_ROOT / "tokenizer_balanced.json"

QUESTION_MAX_LENGTH = 64
COLUMN_MAX_LENGTH = 16


def encode(text: str, tokenizer, max_length: int):
    pad_id = tokenizer.vocab["<PAD>"]

    token_ids = tokenizer.encode_ids(text)[:max_length]
    mask = [1] * len(token_ids)

    padding = max_length - len(token_ids)

    token_ids += [pad_id] * padding
    mask += [0] * padding

    return (
        torch.tensor([token_ids], dtype=torch.long),
        torch.tensor([mask], dtype=torch.long),
    )

def lexical_match_score(context: str, column: str) -> float:
    context_tokens = set(re.findall(r"[A-Za-z0-9]+", context.lower()))
    column_tokens = re.findall(r"[A-Za-z0-9]+", column.lower().replace("_", " "))

    if not column_tokens:
        return 0.0

    matches = sum(token in context_tokens for token in column_tokens)

    return matches / len(column_tokens)


tokenizer = BPETrainer.load(TOKENIZER_PATH)

model = SemanticColumnRanker(
    vocabulary_size=len(tokenizer.vocab),
    embedding_dimension=64,
    hidden_dimension=64,
)

model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()


groups = defaultdict(list)

with open(DEV_PATH, "r", encoding="utf-8") as file:
    for line in file:
        row = json.loads(line)

        key = (row["example_id"], row["table"])
        groups[key].append(row)

weights = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0]

results = {
    weight: {
        "top1": 0,
        "top3": 0,
    }
    for weight in weights
}

total = 0

with torch.no_grad():
    for rows in groups.values():
        ranked = []

        for row in rows:
            question_ids, question_mask = encode(
                row["context"],
                tokenizer,
                QUESTION_MAX_LENGTH,
            )

            column_text = f'{row["table"]} {row["column"]}'

            column_ids, column_mask = encode(
                column_text,
                tokenizer,
                COLUMN_MAX_LENGTH,
            )

            neural_score = model(
                question_ids=question_ids,
                question_mask=question_mask,
                column_ids=column_ids,
                column_mask=column_mask,
            ).item()

            lexical_score = lexical_match_score(
                context=row["context"],
                column=row["column"],
            )

            ranked.append({
                "column": row["column"],
                "label": row["label"],
                "neural_score": neural_score,
                "lexical_score": lexical_score,
            })

        for weight in weights:
            weighted_ranked = sorted(
                ranked,
                key=lambda item: item["neural_score"] + weight * item["lexical_score"],
                reverse=True,
            )

            if weighted_ranked[0]["label"] == 1:
                results[weight]["top1"] += 1

            if any(item["label"] == 1 for item in weighted_ranked[:3]):
                results[weight]["top3"] += 1

        total += 1


print("Ranking groups:", total)

for weight in weights:
    top1 = results[weight]["top1"] / total
    top3 = results[weight]["top3"] / total

    print(
        f"Weight {weight:.1f} | "
        f"Top-1 = {top1:.4f} | "
        f"Top-3 = {top3:.4f}"
    )

## testing
def extract_test_context(question: str, value, window: int = 5) -> str:
    tokens = re.findall(r"[A-Za-z_]+|\d+(?:\.\d+)?|[^\w\s]", question)

    for index, token in enumerate(tokens):
        if re.fullmatch(r"\d+(?:\.\d+)?", token) and float(token) == float(value):
            start = max(0, index - window)
            end = min(len(tokens), index + window + 1)
            return " ".join(tokens[start:end])

    return question


def rank_project_columns(question: str, table: str, columns: list[str], value, weight: float = 3.0):
    context = extract_test_context(question, value)

    ranked = []

    with torch.no_grad():
        for column in columns:
            question_ids, question_mask = encode(
                context,
                tokenizer,
                QUESTION_MAX_LENGTH,
            )

            column_text = f"{table} {column}"

            column_ids, column_mask = encode(
                column_text,
                tokenizer,
                COLUMN_MAX_LENGTH,
            )

            neural_score = model(
                question_ids=question_ids,
                question_mask=question_mask,
                column_ids=column_ids,
                column_mask=column_mask,
            ).item()

            lexical_score = lexical_match_score(
                context=context,
                column=column,
            )

            final_score = neural_score + weight * lexical_score

            ranked.append({
                "column": column,
                "neural": neural_score,
                "lexical": lexical_score,
                "final": final_score,
            })

    ranked.sort(key=lambda item: item["final"], reverse=True)

    return context, ranked

project_tests = [
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
    {
        "question": "show orders above 100",
        "table": "orders",
        "columns": ["id", "product_id", "employee_id", "quantity", "total_amount", "status"],
        "value": 100,
        "expected": "total_amount",
    },
]

print("\nPROJECT SEMANTIC COLUMN TESTS")

passed = 0

for test in project_tests:
    context, ranked = rank_project_columns(
        question=test["question"],
        table=test["table"],
        columns=test["columns"],
        value=test["value"],
        weight=3.0,
    )

    predicted = ranked[0]["column"]
    success = predicted == test["expected"]

    if success:
        passed += 1

    print("\nQuestion :", test["question"])
    print("Context  :", context)
    print("Expected :", test["expected"])
    print("Predicted:", predicted)
    print("Result   :", "PASS" if success else "FAIL")

    for item in ranked:
        print(
            f'{item["column"]:20s} '
            f'neural={item["neural"]:.4f} '
            f'lexical={item["lexical"]:.2f} '
            f'final={item["final"]:.4f}'
        )

print(f"\nProject tests passed: {passed}/{len(project_tests)}")