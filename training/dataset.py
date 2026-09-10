#bpe.py File

from src.tokenizer.bpe import BPETrainer
from text_to_sql.schema_role_encoder import SchemaRoleEncoder

OPERATOR_NONE = 0
OPERATOR_GREATER_THAN = 1
OPERATOR_LESS_THAN = 2
OPERATOR_EQUALS = 3
OPERATOR_IN = 4


def prepare_training_words(corpus: list[str]) -> list[str]:
    if not isinstance(corpus, list):
        raise TypeError("corpus must be a list.")
    if not corpus:
        raise ValueError("corpus cannot be empty.")

    training_words = []
    for sentence in corpus:
        if not isinstance(sentence, str):
            raise TypeError("Every corpus item must be a string.")
        cleaned_sentence = sentence.strip()
        if not cleaned_sentence:
            continue
        training_words.extend(cleaned_sentence.split())
    if not training_words:
        raise ValueError("corpus contains no usable words.")
    return training_words


def train_tokenizer(corpus: list[str],num_merges: int) -> BPETrainer:
    if num_merges < 0:
        raise ValueError("num_merges cannot be negative.")

    training_words = prepare_training_words(corpus)
    tokenizer = BPETrainer(training_words)
    tokenizer.train(num_merges=num_merges)

    return tokenizer


def encode_corpus(corpus: list[str], tokenizer: BPETrainer) -> list[list[int]]:
    if not tokenizer.vocab:
        raise RuntimeError("The tokenizer must be trained before encoding the corpus.")

    bos_id = tokenizer.vocab["<BOS>"]
    eos_id = tokenizer.vocab["<EOS>"]

    encoded_sequences = []

    for sentence in corpus:
        cleaned_sentence = sentence.strip()
        if not cleaned_sentence:
            continue
        sentence_ids = tokenizer.encode_ids(cleaned_sentence)
        complete_ids = ([bos_id] + sentence_ids + [eos_id])
        encoded_sequences.append(complete_ids)

    if not encoded_sequences:
        raise ValueError("No usable sequences were produced.")

    return encoded_sequences


def create_training_example(token_ids: list[int]) -> tuple[list[int], list[int]]:
    if not isinstance(token_ids, list):
        raise TypeError("token_ids must be a list.")

    if len(token_ids) < 2:
        raise ValueError("A sequence must contain at least two tokens.")

    input_ids = token_ids[:-1]
    target_ids = token_ids[1:]

    return input_ids, target_ids


def create_training_dataset(encoded_sequences: list[list[int]],window_size: int, stride: int) -> list[dict[str, list[int]]]:
    if not isinstance(encoded_sequences, list):
        raise TypeError("encoded_sequences must be a list.")

    if not encoded_sequences:
        raise ValueError("encoded_sequences cannot be empty.")

    training_dataset = []

    for token_ids in encoded_sequences:
        windows = create_context_windows(token_ids,window_size,stride)
        for window in windows:
            input_ids, target_ids = create_training_example(window)
            training_dataset.append({"input_ids": input_ids,"target_ids": target_ids})
    return training_dataset


# def build_text_to_sql_training_dataset(corpus: list[dict], schema: dict[str, list[str]], num_merges: int) -> tuple[BPETrainer, list[dict]]:
#     tokenizer_corpus = [example["prompt"] + example["sql"] for example in corpus]
#     tokenizer = train_tokenizer(corpus=tokenizer_corpus, num_merges=num_merges)
#     # role_encoder = SchemaRoleEncoder(tokenizer=tokenizer)
#     #
#     # schema_token_ids, schema_role_ids = role_encoder.encode_schema(schema)
#     training_dataset = []
#
#     for example in corpus:
#         prompt = example["prompt"]
#         sql = example["sql"]
#
#         schema_start = prompt.find("Table ")
#         schema_end = prompt.find("\n\nQuestion:")
#
#         prefix_ids = tokenizer.encode_ids(prompt[:schema_start])
#         suffix_ids = tokenizer.encode_ids(prompt[schema_end:])
#         sql_ids = tokenizer.encode_ids(sql)
#
#         prompt_ids = prefix_ids + schema_token_ids + suffix_ids
#         input_ids = prompt_ids + sql_ids[:-1]
#
#         role_ids = [SchemaRoleEncoder.NORMAL] * len(prefix_ids) + schema_role_ids + [SchemaRoleEncoder.NORMAL] * len(suffix_ids) + [SchemaRoleEncoder.NORMAL] * (len(sql_ids) - 1)
#
#         if len(input_ids) != len(role_ids):
#             raise RuntimeError("input_ids and schema_role_ids are not aligned.")
#
#         training_dataset.append({"input_ids": input_ids, "target_ids": sql_ids, "prompt_length": len(prompt_ids), "schema_role_ids": role_ids})
#
#     return tokenizer, training_dataset

# def build_text_to_sql_training_dataset(corpus: list[dict], schema: dict[str, list[str]], num_merges: int) -> tuple[BPETrainer, list[dict]]:
#     tokenizer_corpus = [example["prompt"] + example["sql"] for example in corpus]
#     tokenizer = train_tokenizer(corpus=tokenizer_corpus, num_merges=num_merges)
#     training_dataset = []
#
#     for example in corpus:
#         prompt = example["prompt"]
#         sql = example["sql"]
#
#         question_start = prompt.find("Question:")
#         question_prompt = prompt[question_start:]
#
#         prompt_ids = tokenizer.encode_ids(question_prompt)
#         sql_ids = tokenizer.encode_ids(sql)
#
#         input_ids = prompt_ids + sql_ids[:-1]
#         role_ids = [SchemaRoleEncoder.NORMAL] * len(input_ids)
#         operator_label = get_operator_label(sql)
#
#         training_dataset.append({
#             "input_ids": input_ids,
#             "target_ids": sql_ids,
#             "prompt_length": len(prompt_ids),
#             "schema_role_ids": role_ids,
#             "operator_label": operator_label,
#         })
#
    # return tokenizer, training_dataset


def build_text_to_sql_training_dataset(corpus: list[dict], tokenizer: BPETrainer) -> list[dict]:
    training_dataset = []
    role_encoder = SchemaRoleEncoder(tokenizer)

    for example in corpus:
        prompt = example["prompt"]
        sql = example["sql"]
        schema = example["schema"]

        question_start = prompt.find("Question:")
        if question_start == -1:
            raise ValueError("Prompt must contain 'Question:'.")

        schema_ids, schema_role_ids = role_encoder.encode_schema(schema)
        question_ids = tokenizer.encode_ids(prompt[question_start:])
        sql_ids = tokenizer.encode_ids(sql)

        prompt_ids = schema_ids + question_ids
        input_ids = prompt_ids + sql_ids[:-1]

        role_ids = (
            schema_role_ids
            + [SchemaRoleEncoder.NORMAL] * len(question_ids)
            + [SchemaRoleEncoder.NORMAL] * (len(sql_ids) - 1))

        if len(input_ids) != len(role_ids):
            raise RuntimeError("input_ids and schema_role_ids are not aligned.")

        training_dataset.append({
            "input_ids": input_ids,
            "target_ids": sql_ids,
            "prompt_length": len(prompt_ids),
            "schema_role_ids": role_ids,
            "operator_label": get_operator_label(sql),
            "where_column": get_where_column(sql, schema),
            "table_name": get_table_name(sql, schema),
            "schema": schema,
        })

    return training_dataset


def create_context_windows(token_ids: list[int], window_size: int, stride: int) -> list[list[int]]:
    if not isinstance(token_ids, list):
        raise TypeError("token_ids must be a list.")

    if window_size < 2:
        raise ValueError("window_size must be at least 2.")

    if stride <= 0:
        raise ValueError("stride must be positive.")

    windows = []
    start = 0
    while start < len(token_ids):
        window = token_ids[start:start + window_size]
        if len(window) >= 2:
            windows.append(window)
        start += stride

    return windows


def create_mini_batches(training_dataset, batch_size):
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    mini_batches = []
    start = 0
    while start < len(training_dataset):
        batch = training_dataset[start:start + batch_size]
        mini_batches.append(batch)
        start += batch_size
    return mini_batches

def get_operator_label(sql: str) -> int:
    if " > " in sql:
        return OPERATOR_GREATER_THAN
    if " < " in sql:
        return OPERATOR_LESS_THAN
    if " IN " in sql:
        return OPERATOR_IN
    if " = " in sql:
        return OPERATOR_EQUALS
    return OPERATOR_NONE

def build_language_training_dataset(corpus: list[str],tokenizer: BPETrainer,window_size: int,stride: int) -> list[dict]:
    eos_id = tokenizer.vocab["<EOS>"]
    token_stream = []
    for sentence in corpus:
        sentence_ids = tokenizer.encode_ids(sentence.strip())
        token_stream.extend(sentence_ids + [eos_id])
    windows = create_context_windows(token_ids=token_stream,window_size=window_size + 1,stride=stride)
    training_dataset = []
    for window in windows:
        input_ids, target_ids = create_training_example(window)
        training_dataset.append({"input_ids": input_ids,"target_ids": target_ids})
    return training_dataset

def get_where_column(sql: str, schema: dict[str, list[str]]) -> str | None:
    upper_sql = sql.upper()

    if " WHERE " not in upper_sql:
        return None

    where_index = upper_sql.rfind(" WHERE ") + len(" WHERE ")
    condition = sql[where_index:].strip()

    candidate_columns = []

    for columns in schema.values():
        candidate_columns.extend(columns)

    # Longest first so similar column names don't conflict.
    candidate_columns.sort(key=len, reverse=True)

    for column in candidate_columns:
        if not condition.startswith(column):
            continue

        remainder = condition[len(column):]

        for operator in [" > ", " < ", " = ", " IN "]:
            if remainder.startswith(operator):
                return column

    return None

def get_table_name(sql: str, schema: dict[str, list[str]]) -> str:
    upper_sql = sql.upper()
    if " FROM " not in upper_sql:
        raise ValueError(f"SQL does not contain FROM: {sql}")
    from_index = upper_sql.rfind(" FROM ") + len(" FROM ")
    after_from = sql[from_index:].strip()
    candidate_tables = sorted(schema.keys(), key=len, reverse=True)
    for table in candidate_tables:
        if after_from == table:
            return table
        if after_from.startswith(table):
            remainder = after_from[len(table):]
            if remainder.startswith(" ") or remainder.startswith(";"):
                return table

    raise ValueError(f"Could not identify table from SQL: {sql}")