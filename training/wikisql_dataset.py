import json


def load_jsonl(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as file:
        return [json.loads(line) for line in file]


def load_wikisql(data_dir: str, limit: int | None = None) -> list[dict]:
    examples = load_jsonl(f"{data_dir}/train.jsonl")
    tables = load_jsonl(f"{data_dir}/train.tables.jsonl")

    table_map = {table["id"]: table for table in tables}
    result = []
    for example in examples:
        example["table"] = table_map[example["table_id"]]
        result.append(example)
        if limit is not None and len(result) >= limit:
            break

    return result


def is_supported_wikisql_example(example: dict) -> bool:
    sql = example["sql"]
    if sql["agg"] not in [0, 3]:
        return False
    if len(sql["conds"]) > 1:
        return False
    for _, operator, _ in sql["conds"]:
        if operator not in [0, 1, 2]:
            return False
    return True

AGGREGATIONS = {
    0: "",
    3: "COUNT",
}

OPERATORS = {
    0: "=",
    1: ">",
    2: "<",
}


def wikisql_to_sql(example: dict) -> str:
    sql = example["sql"]
    headers = example["table"]["header"]

    selected_column = headers[sql["sel"]]
    aggregation = AGGREGATIONS[sql["agg"]]

    if aggregation:
        query = f"SELECT {aggregation}({selected_column}) FROM table"
    else:
        query = f"SELECT {selected_column} FROM table"

    if sql["conds"]:
        column_index, operator_id, value = sql["conds"][0]
        column = headers[column_index]
        operator = OPERATORS[operator_id]
        if isinstance(value, str):
            value = f"'{value}'"
        query += f" WHERE {column} {operator} {value}"
    return query + ";"

def build_wikisql_prompt(example: dict) -> str:
    headers = example["table"]["header"]
    schema_text = "Table table:\n" + "\n".join(headers)

    return (
        f"{schema_text}\n\n"
        f"Question:\n{example['question']}\n\n"
        f"SQL:\n"
    )


def build_wikisql_training_examples(examples: list[dict]) -> list[dict]:
    result = []

    for example in examples:
        if not is_supported_wikisql_example(example):
            continue

        result.append({
            "prompt": build_wikisql_prompt(example),
            "sql": wikisql_to_sql(example),
            "schema": {"table": example["table"]["header"]},
        })

    return result


def build_balanced_wikisql_corpus(examples: list[dict], equals_limit: int = 400) -> list[dict]:
    equals = []
    greater = []
    less = []
    none = []

    for example in build_wikisql_training_examples(examples):
        sql = example["sql"]

        if " > " in sql:
            greater.append(example)
        elif " < " in sql:
            less.append(example)
        elif " = " in sql:
            equals.append(example)
        else:
            none.append(example)

    return equals[:equals_limit] + greater + less + none