from src.tokenizer.bpe import BPETrainer
from text_to_sql.schema_role_encoder import SchemaRoleEncoder
from src.embeddings.schema_role_embedding import SchemaRoleEmbedding


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


def build_text_to_sql_training_dataset(corpus: list[dict], schema: dict[str, list[str]], num_merges: int) -> tuple[BPETrainer, list[dict]]:
    tokenizer_corpus = [example["prompt"] + example["sql"] for example in corpus]
    tokenizer = train_tokenizer(corpus=tokenizer_corpus, num_merges=num_merges)
    role_encoder = SchemaRoleEncoder(tokenizer=tokenizer)

    schema_token_ids, schema_role_ids = role_encoder.encode_schema(schema)
    training_dataset = []

    for example in corpus:
        prompt = example["prompt"]
        sql = example["sql"]

        schema_start = prompt.find("Table ")
        schema_end = prompt.find("\n\nQuestion:")

        prefix_ids = tokenizer.encode_ids(prompt[:schema_start])
        suffix_ids = tokenizer.encode_ids(prompt[schema_end:])
        sql_ids = tokenizer.encode_ids(sql)

        prompt_ids = prefix_ids + schema_token_ids + suffix_ids
        input_ids = prompt_ids + sql_ids[:-1]

        role_ids = [SchemaRoleEmbedding.NORMAL] * len(prefix_ids) + schema_role_ids + [SchemaRoleEmbedding.NORMAL] * len(suffix_ids) + [SchemaRoleEmbedding.NORMAL] * (len(sql_ids) - 1)

        if len(input_ids) != len(role_ids):
            raise RuntimeError("input_ids and schema_role_ids are not aligned.")

        training_dataset.append({"input_ids": input_ids, "target_ids": sql_ids, "prompt_length": len(prompt_ids), "schema_role_ids": role_ids})

    return tokenizer, training_dataset


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