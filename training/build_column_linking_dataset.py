import json
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SPIDER_DIR = PROJECT_ROOT / "data" / "spider"
OUTPUT_DIR = PROJECT_ROOT / "data" / "column_linking"

TRAIN_PATH = SPIDER_DIR / "train_spider.json"
TRAIN_OTHERS_PATH = SPIDER_DIR / "train_others.json"
DEV_PATH = SPIDER_DIR / "dev.json"
TABLES_PATH = SPIDER_DIR / "tables.json"


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def extract_where_column_ids(example: dict) -> list[int]:
    column_ids = []

    for condition in example["sql"]["where"]:
        if isinstance(condition, str):
            continue

        value_unit = condition[2]
        column_unit = value_unit[1]
        column_id = column_unit[1]

        if column_id not in column_ids:
            column_ids.append(column_id)

    return column_ids


def build_where_column_training_rows(example: dict, schema: dict, example_id: int) -> list[dict]:
    condition = extract_simple_numeric_where(example)

    if condition is None:
        return []

    positive_column_id, value = condition

    context = extract_condition_context(
        question=example["question"],
        value=value,
    )

    if context is None:
        return []

    column_names = schema["column_names_original"]
    table_names = schema["table_names_original"]

    table_index = column_names[positive_column_id][0]

    if table_index == -1:
        return []

    rows = []

    for column_id, (candidate_table_index, column_name) in enumerate(column_names):
        if candidate_table_index != table_index:
            continue

        if column_name == "*":
            continue

        rows.append({
            "example_id": example_id,
            "db_id": example["db_id"],
            "question": example["question"],
            "context": context,
            "table": table_names[table_index],
            "column": column_name,
            "label": 1 if column_id == positive_column_id else 0,
        })

    return rows

def extract_condition_context(question: str, value, window: int = 5):
    tokens = re.findall(r"[A-Za-z_]+|\d+(?:\.\d+)?|[^\w\s]", question)

    value_index = None

    for index, token in enumerate(tokens):
        if not re.fullmatch(r"\d+(?:\.\d+)?", token):
            continue

        if float(token) == float(value):
            value_index = index
            break

    if value_index is None:
        return None

    start = max(0, value_index - window)
    end = min(len(tokens), value_index + window + 1)

    return " ".join(tokens[start:end])

def extract_simple_numeric_where(example: dict):
    conditions = [
        condition
        for condition in example["sql"]["where"]
        if not isinstance(condition, str)
    ]

    if len(conditions) != 1:
        return None

    condition = conditions[0]

    operator_id = condition[1]

    # Spider: 2 =, 3 >, 4 <
    if operator_id not in (2, 3, 4):
        return None

    value = condition[3]

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None

    value_unit = condition[2]
    column_unit = value_unit[1]
    column_id = column_unit[1]

    return column_id, value


def build_dataset(examples: list[dict], schemas: dict[str, dict]) -> list[dict]:
    rows = []

    for example_id, example in enumerate(examples):
        if not example["sql"]["where"]:
            continue

        schema = schemas[example["db_id"]]
        rows.extend(build_where_column_training_rows(example, schema, example_id))

    return rows


def save_jsonl(rows: list[dict], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    train_spider = load_json(TRAIN_PATH)
    train_others = load_json(TRAIN_OTHERS_PATH)
    dev_examples = load_json(DEV_PATH)
    tables_data = load_json(TABLES_PATH)

    schemas = {
        schema["db_id"]: schema
        for schema in tables_data
    }

    train_examples = train_spider + train_others

    train_rows = build_dataset(train_examples, schemas)
    dev_rows = build_dataset(dev_examples, schemas)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    save_jsonl(train_rows, OUTPUT_DIR / "train.jsonl")
    save_jsonl(dev_rows, OUTPUT_DIR / "dev.jsonl")

    print("TRAIN")
    print("Spider examples:", len(train_examples))
    print("Column-linking rows:", len(train_rows))
    print("Positive:", sum(row["label"] == 1 for row in train_rows))
    print("Negative:", sum(row["label"] == 0 for row in train_rows))

    print("\nDEV")
    print("Spider examples:", len(dev_examples))
    print("Column-linking rows:", len(dev_rows))
    print("Positive:", sum(row["label"] == 1 for row in dev_rows))
    print("Negative:", sum(row["label"] == 0 for row in dev_rows))