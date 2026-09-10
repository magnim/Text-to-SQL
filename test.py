import sqlite3
import random
import torch

SEED = 42

random.seed(SEED)
torch.manual_seed(SEED)
from pytorch_impl.tiny_gpt import TinyGPT
from training.dataset import (build_text_to_sql_training_dataset,train_tokenizer,
                              # encode_corpus, create_training_dataset,
                              build_language_training_dataset)
from pytorch_impl.train_tiny_gpt import train_tiny_gpt

from text_to_sql.schema_encoder import SchemaEncoder
from text_to_sql.schema_role_encoder import SchemaRoleEncoder
from text_to_sql.prompt_builder import PromptBuilder
from training.text_to_sql_dataset import build_text_to_sql_corpus
from text_to_sql.text_to_sql_pipeline import TextToSQLPipeline
from text_to_sql.sql_evaluator import SQLEvaluator
from src.tokenizer.bpe import BPETrainer


from training.language_corpus import load_language_corpus
from training.pretrain_tiny_gpt import pretrain_tiny_gpt
from training.wikisql_dataset import load_wikisql, is_supported_wikisql_example, wikisql_to_sql, build_wikisql_training_examples,build_balanced_wikisql_corpus
from pytorch_impl.train_column_head import train_column_head
from pytorch_impl.train_table_head import train_table_head


# wikisql = load_wikisql(limit=5000,data_dir="training/data/data")
# wikisql_corpus = build_wikisql_training_examples(wikisql)
#
# print("Existing examples:", len(corpus))
# print("WikiSQL examples:", len(wikisql_corpus))


# ============================================================
# DATABASE SCHEMA
# ============================================================

schema = {
    "employees": [
        "id",
        "name",
        "department",
        "salary",
        "age",
        "years_experience",
    ],
    "products": [
        "id",
        "name",
        "category",
        "price",
        "stock",
        "rating",
    ],
    "orders": [
        "id",
        "product_id",
        "employee_id",
        "quantity",
        "total_amount",
        "status",
    ],
}


# ============================================================
# SQLITE TEST DATABASE
# ============================================================

connection = sqlite3.connect(":memory:")
cursor = connection.cursor()

cursor.execute("""CREATE TABLE employees (
    id INTEGER,
    name TEXT,
    department TEXT,
    salary REAL,
    age INTEGER,
    years_experience INTEGER
)
""")

cursor.execute("""CREATE TABLE products (
    id INTEGER,
    name TEXT,
    category TEXT,
    price REAL,
    stock INTEGER,
    rating REAL
)
""")

cursor.execute("""CREATE TABLE orders (
    id INTEGER,
    product_id INTEGER,
    employee_id INTEGER,
    quantity INTEGER,
    total_amount REAL,
    status TEXT
)
""")


# ============================================================
# SAMPLE DATA
# ============================================================

cursor.executemany(
    "INSERT INTO employees VALUES (?, ?, ?, ?, ?, ?)",
    [
        (1, "Alice", "Engineering", 75000, 30, 6),
        (2, "Bob", "Sales", 55000, 28, 4),
        (3, "Charlie", "Engineering", 90000, 35, 10),
        (4, "David", "HR", 50000, 26, 3),
        (5, "Emma", "Sales", 65000, 31, 7),
    ],
)

cursor.executemany(
    "INSERT INTO products VALUES (?, ?, ?, ?, ?, ?)",
    [
        (1, "Laptop", "Electronics", 900, 20, 4.7),
        (2, "Mouse", "Electronics", 25, 100, 4.3),
        (3, "Chair", "Furniture", 120, 30, 4.1),
        (4, "Keyboard", "Electronics", 45, 80, 4.5),
        (5, "Desk", "Furniture", 250, 15, 4.4),
    ],
)

cursor.executemany(
    "INSERT INTO orders VALUES (?, ?, ?, ?, ?, ?)",
    [
        (1, 1, 1, 1, 900, "completed"),
        (2, 2, 2, 2, 50, "pending"),
        (3, 4, 3, 1, 45, "completed"),
        (4, 3, 1, 1, 120, "cancelled"),
        (5, 5, 5, 1, 250, "completed"),
    ],
)

connection.commit()


# ============================================================
# BUILD TEXT-to-SQL CORPUS
# ============================================================

schema_encoder = SchemaEncoder()
schema_text = schema_encoder.encode_schema(schema)

prompt_builder = PromptBuilder()

corpus = build_text_to_sql_corpus(
    schema_text=schema_text,
    prompt_builder=prompt_builder)

for example in corpus:
    example["schema"] = schema

wikisql = load_wikisql(
    limit=None,
    data_dir="training/data/data",
)

wikisql_corpus = build_balanced_wikisql_corpus(
    wikisql,
    equals_limit=400
)

# tiny_wikisql_corpus = wikisql_corpus[:5]
# combined_corpus = tiny_wikisql_corpus
combined_corpus = corpus + wikisql_corpus
print("Tiny WikiSQL examples:", len(combined_corpus))

counts = {
    "NONE": 0,
    ">": 0,
    "<": 0,
    "=": 0,
}

for example in wikisql_corpus:
    sql = example["sql"]

    if " > " in sql:
        counts[">"] += 1
    elif " < " in sql:
        counts["<"] += 1
    elif " = " in sql:
        counts["="] += 1
    else:
        counts["NONE"] += 1

print("Supported WikiSQL:", len(wikisql_corpus))
print("NONE:", counts["NONE"])
print(">   :", counts[">"])
print("<   :", counts["<"])
print("=   :", counts["="])
combined_corpus = corpus + wikisql_corpus
# tiny_wikisql_corpus = wikisql_corpus[:5]
# combined_corpus = tiny_wikisql_corpus
#
# print("Tiny WikiSQL examples:", len(tiny_wikisql_corpus))
print("Combined examples:", len(combined_corpus))

print("WikiSQL balanced examples:", len(wikisql_corpus))
print("Combined examples:", len(combined_corpus))


print("Existing examples:", len(corpus))
print("WikiSQL examples:", len(wikisql_corpus))
print("Combined examples:", len(combined_corpus))



# ============================================================
# TOKENIZER + TRAINING DATASET
# ============================================================

# tokenizer, training_dataset = build_text_to_sql_training_dataset(
#     corpus=corpus,schema=schema,
#     num_merges=100,
#     # window_size=13,
#     # stride=6,
# )

# tokenizer_corpus = [example["prompt"] + example["sql"] for example in corpus]
# tokenizer = train_tokenizer(tokenizer_corpus, num_merges=100)

LANGUAGE_CORPUS = load_language_corpus(limit=5000)
print("Language rows:", len(LANGUAGE_CORPUS))

LOAD_MODEL = True
# TARGETED_ORDERBY_FINETUNE = False
# TARGETED_ORDERS_FINETUNE = False
# TARGETED_ORDERBY_LANGUAGE_FINETUNE = False
# TARGETED_ORDERBY_BALANCED_FINETUNE = True

# MODEL_PATH = "tiny_gpt_text_to_sql_core_sql.pt"
# TOKENIZER_PATH = "tokenizer.json"

# MODEL_PATH = "tiny_gpt_text_to_sql_balanced.pt"
# MODEL_PATH = "tiny_gpt_text_to_sql_balanced_orderby.pt"

# MODEL_PATH = "tiny_gpt_text_to_sql_final_candidate.pt"
# MODEL_PATH = "tiny_gpt_text_to_sql_with_in.pt"
# MODEL_PATH = "tiny_gpt_text_to_sql_with_in_v2.pt"
# MODEL_PATH = "tiny_gpt_text_to_sql_with_in_v3.pt"
MODEL_PATH = "tiny_gpt_text_to_sql_with_in_v4.pt"
TOKENIZER_PATH = "tokenizer_balanced.json"

text_to_sql_corpus = [example["prompt"] + example["sql"] for example in combined_corpus]
tokenizer_corpus = LANGUAGE_CORPUS + text_to_sql_corpus

if LOAD_MODEL:
    print("\n===== LOADING SAVED TOKENIZER =====")
    tokenizer = BPETrainer.load(TOKENIZER_PATH)
    print("Tokenizer loaded.")
else:
    print("\n===== TRAINING TOKENIZER =====")
    tokenizer = train_tokenizer(tokenizer_corpus, num_merges=1000)
    tokenizer.save(TOKENIZER_PATH)
    print("Tokenizer saved.")
# loaded_tokenizer = BPETrainer.load(TOKENIZER_PATH)
#
# test_text = "SELECT * FROM employees WHERE salary > 60000;"
#
# original_ids = tokenizer.encode_ids(test_text)
# loaded_ids = loaded_tokenizer.encode_ids(test_text)
#
# print("\n===== TOKENIZER SAVE/LOAD CHECK =====")
# print("Original:", original_ids)
# print("Loaded  :", loaded_ids)
# print("Match   :", original_ids == loaded_ids)
# print("Decoded :", loaded_tokenizer.decode_ids(loaded_ids))
# print("=====================================\n")


training_dataset = build_text_to_sql_training_dataset(
    corpus=combined_corpus,
    tokenizer=tokenizer,
)

print("\n===== CUSTOM TABLE DISTRIBUTION =====")

custom_table_counts = {
    "employees": 0,
    "products": 0,
    "orders": 0,
}

for example in training_dataset[:len(corpus)]:
    table_name = example["table_name"]

    if table_name in custom_table_counts:
        custom_table_counts[table_name] += 1

for table, count in custom_table_counts.items():
    print(f"{table:12}: {count}")

print("Total       :", sum(custom_table_counts.values()))
print("=====================================\n")

print("\n===== MULTI-TABLE TRAINING CHECK =====")

multi_table_examples = 0
single_table_examples = 0

for example in training_dataset:
    if len(example["schema"]) > 1:
        multi_table_examples += 1
    else:
        single_table_examples += 1

print("Multi-table examples :", multi_table_examples)
print("Single-table examples:", single_table_examples)
print("Total examples       :", len(training_dataset))

print("======================================\n")
print("\n===== TABLE LABEL CHECK =====")

for index in [0, 1, 2, 100, 500, 1000]:
    example = training_dataset[index]

    print("Example:", index)
    print("Table  :", example["table_name"])
    print("Schema :", list(example["schema"].keys()))
    print()

print("=============================\n")
invalid_table_labels = 0

for index, example in enumerate(training_dataset):
    table_name = example["table_name"]
    example_schema = example["schema"]

    if table_name not in example_schema:
        invalid_table_labels += 1

        print("Invalid table label")
        print("Index :", index)
        print("Table :", table_name)
        print("Schema:", list(example_schema.keys()))
        print()

print("Invalid table labels:", invalid_table_labels)
print("\n===== WHERE COLUMN LABEL CHECK =====")

missing_where_columns = 0

for example in training_dataset:
    if example["operator_label"] != 0 and example["where_column"] is None:
        missing_where_columns += 1

print("Missing WHERE columns:", missing_where_columns)

check_indices = [0, 64, 76, 80, 85, 88, 100]

for index in check_indices:
    print(
        "SQL:",
        combined_corpus[index]["sql"]
    )
    print(
        "WHERE column:",
        training_dataset[index]["where_column"]
    )
    print()

print("====================================\n")

print("\n===== MISSING WHERE COLUMN EXAMPLES =====")

for index, example in enumerate(training_dataset):
    if example["operator_label"] != 0 and example["where_column"] is None:
        print("Index :", index)
        print("SQL   :", combined_corpus[index]["sql"])
        print("Schema:", example["schema"])
        print()

print("=========================================\n")


# test_tokens = [
#     "SELECT",
#     "FROM",
#     "WHERE",
#     "COUNT",
#     "AVG",
#     "DISTINCT",
#     "employees",
#     "products",
#     "orders",
#     "salary",
#     "age",
#     "price",
#     "status",
#     "total_amount",
#     "SELECT * FROM employees;",
# ]
#
# print("\n===== TOKENIZER DIAGNOSTIC =====")
#
# for text in test_tokens:
#     tokens = tokenizer.encode(text)
#     ids = tokenizer.encode_ids(text)
#
#     print(f"{text:30} -> {tokens}")
#     print(f"{'':30} -> {ids}")
#
# print("Vocabulary size:", len(tokenizer.vocab))
# print("===============================\n")


# encoded_language = encode_corpus(LANGUAGE_CORPUS, tokenizer)
#
# language_dataset = create_training_dataset(
#     encoded_sequences=encoded_language,
#     window_size=32,
#     stride=32
# )

language_dataset = build_language_training_dataset(
    corpus=LANGUAGE_CORPUS,
    tokenizer=tokenizer,
    window_size=32,
    stride=32
)
print("Language sentences:", len(LANGUAGE_CORPUS))
print("Language training samples:", len(language_dataset))
# for table in ["employees", "products", "orders"]:
#     ids = tokenizer.encode_ids(table)
#     print(table, "IDs:", ids, "Length:", len(ids))
#
# for sql in [
#     "SELECT * FROM employees;",
#     "SELECT * FROM products;",
#     "SELECT * FROM orders;",
# ]:
#     ids = tokenizer.encode_ids(sql)
#     print(sql, "Length:", len(ids), "IDs:", ids)

# print("Vocabulary size:", len(tokenizer.vocab))
# print("Training samples:", len(training_dataset))

print("\n===== OPERATOR LABEL CHECK =====")

for i, example in enumerate(combined_corpus):
    if i >= 10:
        break

    print(
        example["sql"],
        "->",
        training_dataset[i]["operator_label"]
    )

print("================================\n")
print("\n===== OPERATOR LABEL COUNTS =====")

counts = {}

for example in training_dataset:
    label = example["operator_label"]
    counts[label] = counts.get(label, 0) + 1

print("0 NONE :", counts.get(0, 0))
print("1 >    :", counts.get(1, 0))
print("2 <    :", counts.get(2, 0))
print("3 =    :", counts.get(3, 0))
print("4 IN   :", counts.get(4, 0))

print("=================================\n")

print("\n===== IN OPERATOR CHECK =====")

for index, example in enumerate(combined_corpus):
    if " IN " in example["sql"]:
        print(
            example["sql"],
            "->",
            training_dataset[index]["operator_label"]
        )

print("=============================\n")
print("\n===== IN OPERATOR CHECK =====")

for index, example in enumerate(combined_corpus):
    if " IN " in example["sql"]:
        print(
            example["sql"],
            "->",
            training_dataset[index]["operator_label"]
        )

print("=============================\n")

print("\n===== SQL TOKENIZATION CHECK =====")

for table in ["employees", "products", "orders"]:
    sql = f"SELECT * FROM {table};"

    full_ids = tokenizer.encode_ids(sql)
    piece_ids = tokenizer.encode_ids("SELECT ") + tokenizer.encode_ids("*") + tokenizer.encode_ids(" FROM ") + tokenizer.encode_ids(table) + tokenizer.encode_ids(";")

    print("\nSQL:", sql)
    print("FULL :", full_ids)
    print("PIECE:", piece_ids)
    print("MATCH:", full_ids == piece_ids)

for table in ["employees", "products", "orders"]:
    sql = f"SELECT COUNT(*) FROM {table};"

    full_ids = tokenizer.encode_ids(sql)
    piece_ids = tokenizer.encode_ids("SELECT ") + tokenizer.encode_ids("COUNT") + tokenizer.encode_ids("(") + tokenizer.encode_ids("*") + tokenizer.encode_ids(")") + tokenizer.encode_ids(" FROM ") + tokenizer.encode_ids(table) + tokenizer.encode_ids(";")

    print("\nSQL:", sql)
    print("FULL :", full_ids)
    print("PIECE:", piece_ids)
    print("MATCH:", full_ids == piece_ids)

sql = "SELECT * FROM employees WHERE age IN (25, 30);"

full_ids = tokenizer.encode_ids(sql)

piece_ids = (
        tokenizer.encode_ids("SELECT ")
        + tokenizer.encode_ids("*")
        + tokenizer.encode_ids(" FROM ")
        + tokenizer.encode_ids("employees")
        + tokenizer.encode_ids(" WHERE ")
        + tokenizer.encode_ids("age")
        + tokenizer.encode_ids(" IN ")
        + tokenizer.encode_ids("(")
        + tokenizer.encode_ids("25")
        + tokenizer.encode_ids(",")
        + tokenizer.encode_ids(" ")
        + tokenizer.encode_ids("30")
        + tokenizer.encode_ids(")")
        + tokenizer.encode_ids(";") )

print("\nIN SQL:", sql)
print("FULL :", full_ids)
print("PIECE:", piece_ids)
print("MATCH:", full_ids == piece_ids)

print("==================================\n")

print("\n===== SCHEMA ROLE CHECK =====")

for index in [0, 1]:
    example = training_dataset[index]

    print("Example:", index)
    print("NORMAL:", example["schema_role_ids"].count(0))
    print("TABLE :", example["schema_role_ids"].count(1))
    print("COLUMN:", example["schema_role_ids"].count(2))
    print()

print("=============================\n")



# ============================================================
# MODEL
# ============================================================
max_length = max(len(example["input_ids"]) for example in training_dataset)
print(f"Longest training sequence  : {max_length}")
# maximum_sequence_length = max_length + 32

model = TinyGPT(
    vocabulary_size=len(tokenizer.vocab),
    embedding_dimension=64,
    maximum_sequence_length=512,
    number_of_heads=4,
    hidden_dimension=128,
    number_of_layers=4,
)
# LOAD_MODEL = True
# MODEL_PATH = "tiny_gpt_text_to_sql.pt"
# MODEL_PATH = "tiny_gpt_text_to_sql_column_frozen.pt"
# MODEL_PATH = "tiny_gpt_text_to_sql_table_frozen.pt"
# COLUMN_MODEL_PATH = "tiny_gpt_text_to_sql_column_frozen.pt"
# TABLE_MODEL_PATH = "tiny_gpt_text_to_sql_table_frozen.pt"

# MODEL_PATH = "tiny_gpt_text_to_sql_core_sql.pt"
# TOKENIZER_PATH = "tokenizer_core_sql.json"
# LOAD_MODEL = False
sample = training_dataset[0]

operator_logits = model.predict_operator(sample["input_ids"],sample["schema_role_ids"])

print("Operator logits shape:", operator_logits.shape)

# ============================================================
# TRAIN MODEL
# ============================================================

# epoch_losses = train_tiny_gpt(
#     model=model,
#     training_dataset=training_dataset,
#     epochs=50,
#     learning_rate=0.001,
#     batch_size=2,
# )
# print("Language rows:", len(LANGUAGE_CORPUS))
# print("Vocabulary size:", len(tokenizer.vocab))


language_dataset = language_dataset[:20000]
print("Language training samples:", len(language_dataset))


print("\n===== COMBINED DATASET CHECK =====")
print("Existing examples:", len(corpus))
print("WikiSQL examples:", len(wikisql_corpus))
print("Combined examples:", len(combined_corpus))
print("Vocabulary size:", len(tokenizer.vocab))
print("Training samples:", len(training_dataset))

max_length = max(len(example["input_ids"]) for example in training_dataset)
print("Longest training sequence:", max_length)

counts = {}
for example in training_dataset:
    label = example["operator_label"]
    counts[label] = counts.get(label, 0) + 1

print("NONE:", counts.get(0, 0))
print(">   :", counts.get(1, 0))
print("<   :", counts.get(2, 0))
print("=   :", counts.get(3, 0))
print("IN  :", counts.get(4, 0))
print("==================================")




# print("\n===== LANGUAGE PRETRAINING =====")
#
# pretrain_tiny_gpt(
#     model=model,
#     training_dataset=language_dataset,
#     epochs=5,
#     learning_rate=0.001,
#     batch_size=4
# )
#
# print("\n===== LOSS ALIGNMENT CHECK =====")
#
# example = training_dataset[0]
#
# input_ids = example["input_ids"]
# target_ids = example["target_ids"]
# prompt_length = example["prompt_length"]
#
# print("Input length :", len(input_ids))
# print("Prompt length:", prompt_length)
# print("Target length:", len(target_ids))
#
# print("\nPrompt:")
# print(tokenizer.decode_ids(input_ids[:prompt_length]))
#
# print("\nSQL inside input:")
# print(tokenizer.decode_ids(input_ids[prompt_length:]))
#
# print("\nTarget SQL:")
# print(tokenizer.decode_ids(target_ids))
#
# print("\nExpected relationship:")
# print("Input SQL   :", tokenizer.decode_ids(input_ids[prompt_length:]))
# print("Target[:-1] :", tokenizer.decode_ids(target_ids[:-1]))
#
# print("\nLengths:")
# print("Logit positions:", len(input_ids) - (prompt_length - 1))
# print("Target tokens  :", len(target_ids))
#
# print("============================\n")
print("\n===== WHERE COLUMN DISTRIBUTION =====")

counts = {
    "salary >": 0,
    "salary <": 0,
    "salary =": 0,
    "age >": 0,
    "age <": 0,
    "age =": 0,
}

for example, source in zip(training_dataset, combined_corpus):
    where_column = example["where_column"]
    sql = source["sql"]

    if where_column == "salary":
        if " > " in sql:
            counts["salary >"] += 1
        elif " < " in sql:
            counts["salary <"] += 1
        elif " = " in sql:
            counts["salary ="] += 1

    elif where_column == "age":
        if " > " in sql:
            counts["age >"] += 1
        elif " < " in sql:
            counts["age <"] += 1
        elif " = " in sql:
            counts["age ="] += 1

for label, count in counts.items():
    print(f"{label:10}: {count}")

salary_total = sum(count for label, count in counts.items() if label.startswith("salary"))
age_total = sum(count for label, count in counts.items() if label.startswith("age"))

print("Salary total:", salary_total)
print("Age total   :", age_total)

print("=====================================\n")


if LOAD_MODEL:
    print("\n===== LOADING SAVED MODEL =====")

    load_result = model.load_state_dict(
        torch.load(MODEL_PATH, map_location="cpu"),
        strict=False
    )

    print("Missing keys:", load_result.missing_keys)
    print("Unexpected keys:", load_result.unexpected_keys)

    model.eval()
    print("Model loaded.")
    # print("\n===== V4 EQUALITY CORRECTION FINE-TUNING =====")
    #
    # custom_count = len(corpus)
    #
    # custom_training = training_dataset[:custom_count]
    # wikisql_training = training_dataset[custom_count:]
    #
    # fine_tuning_dataset = custom_training * 5 + wikisql_training
    # random.shuffle(fine_tuning_dataset)
    #
    # print("Custom original :", len(custom_training))
    # print("Custom x5       :", len(custom_training) * 5)
    # print("WikiSQL         :", len(wikisql_training))
    # print("Total           :", len(fine_tuning_dataset))
    #
    # train_tiny_gpt(
    #     model=model,
    #     tokenizer=tokenizer,
    #     training_dataset=fine_tuning_dataset,
    #     epochs=2,
    #     learning_rate=0.0001,
    #     batch_size=4,
    # )
    #
    # torch.save(
    #     model.state_dict(),
    #     "tiny_gpt_text_to_sql_with_in_v4.pt"
    # )
    #
    # model.eval()
    # print("V4 model saved.")
    # exit()

    # print("\n===== IN V3 FINE-TUNING =====")
    #
    # custom_count = len(corpus)
    #
    # custom_training = training_dataset[:custom_count]
    # wikisql_training = training_dataset[custom_count:]
    #
    # fine_tuning_dataset = custom_training * 5 + wikisql_training
    # random.shuffle(fine_tuning_dataset)
    #
    # print("Custom original :", len(custom_training))
    # print("Custom x5       :", len(custom_training) * 5)
    # print("WikiSQL         :", len(wikisql_training))
    # print("Total           :", len(fine_tuning_dataset))
    #
    # train_tiny_gpt(
    #     model=model,
    #     tokenizer=tokenizer,
    #     training_dataset=fine_tuning_dataset,
    #     epochs=5,
    #     learning_rate=0.0001,
    #     batch_size=4
    # )
    #
    # torch.save(
    #     model.state_dict(),
    #     "tiny_gpt_text_to_sql_with_in_v3.pt"
    # )
    #
    # model.eval()
    # print("IN V3 model saved.")
    # print("\n===== IN CAPABILITY FINE-TUNING =====")
    #
    # custom_count = len(corpus)
    # custom_training = training_dataset[:custom_count]
    # wikisql_training = training_dataset[custom_count:]
    #
    # fine_tuning_dataset = custom_training * 5 + wikisql_training
    # random.shuffle(fine_tuning_dataset)
    #
    # print("Custom original :", len(custom_training))
    # print("Custom x5       :", len(custom_training) * 5)
    # print("WikiSQL         :", len(wikisql_training))
    # print("Total           :", len(fine_tuning_dataset))
    #
    # train_tiny_gpt(
    #     model=model,
    #     tokenizer=tokenizer,
    #     training_dataset=fine_tuning_dataset,
    #     epochs=5,
    #     learning_rate=0.0001,
    #     batch_size=4
    # )
    #
    # torch.save(model.state_dict(), "tiny_gpt_text_to_sql_with_in_v2.pt")
    #
    # model.eval()
    # print("IN-capability model saved.")
    # if TARGETED_ORDERBY_LANGUAGE_FINETUNE:
    #     print("\n===== TARGETED ORDER BY LANGUAGE FINE-TUNING =====")
    #
    #     custom_count = len(corpus)
    #     custom_training = training_dataset[:custom_count]
    #
    #     new_orderby_questions = {
    #         "sort employees from low salary to high salary",
    #         "order employees from least paid to most paid",
    #         "sort products from lower price to higher price",
    #         "order products from least costly to most costly",
    #         "sort orders from lower quantity to higher quantity",
    #         "order orders from fewer items to more items",
    #         "sort employees from high salary to low salary",
    #         "sort products from higher price to lower price",
    #         "sort orders from higher quantity to lower quantity",
    #     }
    #
    #     new_orderby_training = [
    #         training_dataset[index]
    #         for index, example in enumerate(combined_corpus[:custom_count])
    #         if example["question"] in new_orderby_questions
    #     ]
    #
    #     targeted_training = custom_training + new_orderby_training * 3
    #     random.shuffle(targeted_training)
    #
    #     print("Custom examples          :", len(custom_training))
    #     print("New ORDER BY examples    :", len(new_orderby_training))
    #     print("Targeted total           :", len(targeted_training))
    #
    #     train_tiny_gpt(
    #         model=model,
    #         tokenizer=tokenizer,
    #         training_dataset=targeted_training,
    #         epochs=5,
    #         learning_rate=0.0001,
    #         batch_size=4
    #     )
    #
    #     torch.save(
    #         model.state_dict(),
    #         "tiny_gpt_text_to_sql_final_orderby_language.pt"
    #     )
    #
    #     model.eval()
    #     print("ORDER BY language model saved.")
    # # if TARGETED_ORDERBY_FINETUNE:
    # #     print("\n===== TARGETED ORDER BY FINE-TUNING =====")
    # #
    # #     custom_count = len(corpus)
    # #     custom_training = training_dataset[:custom_count]
    # #
    # #     order_by_training = [
    # #         training_dataset[index]
    # #         for index, example in enumerate(combined_corpus[:custom_count])
    # #         if "ORDER BY" in example["sql"].upper()
    # #     ]
    # #
    # #     targeted_training = custom_training + order_by_training * 4
    # #     random.shuffle(targeted_training)
    # #
    # #     print("Custom examples  :", len(custom_training))
    # #     print("ORDER BY examples:", len(order_by_training))
    # #     print("Targeted total   :", len(targeted_training))
    # #
    # #     train_tiny_gpt(
    # #         model=model,
    # #         tokenizer=tokenizer,
    # #         training_dataset=targeted_training,
    # #         epochs=10,
    # #         learning_rate=0.0001,
    # #         batch_size=4
    # #     )
    # #
    # #     torch.save(
    # #         model.state_dict(),
    # #         "tiny_gpt_text_to_sql_balanced_orderby.pt"
    # #     )
    # #
    # #     model.eval()
    # #     print("Targeted ORDER BY model saved.")
    # if TARGETED_ORDERS_FINETUNE:
    #     print("\n===== TARGETED ORDERS FINE-TUNING =====")
    #
    #     custom_count = len(corpus)
    #     custom_training = training_dataset[:custom_count]
    #
    #     orders_numeric_training = [
    #         training_dataset[index]
    #         for index, example in enumerate(combined_corpus[:custom_count])
    #         if "orders" in example["sql"].lower()
    #            and (
    #                    "quantity" in example["sql"].lower()
    #                    or "total_amount" in example["sql"].lower()
    #            )
    #     ]
    #
    #     targeted_training = custom_training + orders_numeric_training * 3
    #     random.shuffle(targeted_training)
    #
    #     print("Custom examples          :", len(custom_training))
    #     print("Orders numeric examples  :", len(orders_numeric_training))
    #     print("Targeted total           :", len(targeted_training))
    #
    #     train_tiny_gpt(
    #         model=model,
    #         tokenizer=tokenizer,
    #         training_dataset=targeted_training,
    #         epochs=10,
    #         learning_rate=0.0001,
    #         batch_size=4
    #     )
    #
    #     torch.save(
    #         model.state_dict(),
    #         "tiny_gpt_text_to_sql_final_candidate.pt"
    #     )
    #
    #     model.eval()
    #     print("Targeted orders model saved.")

else:
    print("\n===== LANGUAGE PRETRAINING =====")

    pretrain_tiny_gpt(
        model=model,
        training_dataset=language_dataset,
        epochs=5,
        learning_rate=0.001,
        batch_size=4
    )

    print("\n===== TEXT-TO-SQL FINE-TUNING =====")

    custom_count = len(corpus)

    custom_training = training_dataset[:custom_count]
    wikisql_training = training_dataset[custom_count:]

    balanced_training_dataset = custom_training * 5 + wikisql_training

    random.shuffle(balanced_training_dataset)

    print("\n===== BALANCED FINE-TUNING DATASET =====")
    print("Custom original :", len(custom_training))
    print("Custom x5       :", len(custom_training) * 5)
    print("WikiSQL         :", len(wikisql_training))
    print("Total           :", len(balanced_training_dataset))
    print("========================================\n")

    train_tiny_gpt(
        model=model,
        tokenizer=tokenizer,
        training_dataset=balanced_training_dataset,
        epochs=20,
        learning_rate=0.001,
        batch_size=4
    )

    torch.save(model.state_dict(), MODEL_PATH)
    print("Model saved.")

# print("\n===== COLUMN HEAD TRAINING =====")
#
# train_column_head(
#     model=model,
#     tokenizer=tokenizer,
#     training_dataset=training_dataset,
#     epochs=10,
#     learning_rate=0.001,
# )
#
# torch.save(model.state_dict(),"tiny_gpt_text_to_sql_column_frozen.pt")
#
# print("Column-head model saved.")
# print("\n===== TABLE HEAD TRAINING =====")
#
# train_table_head(
#     model=model,
#     tokenizer=tokenizer,
#     training_dataset=training_dataset,
#     epochs=10,
#     learning_rate=0.001,
# )
#
# torch.save(model.state_dict(), TABLE_MODEL_PATH)
#
# print("Table-head model saved.")
# print("\n===== TABLE HEAD ACCURACY TEST =====")
#
# table_test_cases = [
#     ("show all employees", "employees"),
#     ("find employees earning more than 60000", "employees"),
#     ("find employees older than 30", "employees"),
#
#     ("show all products", "products"),
#     ("show products cheaper than 50", "products"),
#     ("what is the average product price", "products"),
#
#     ("show all orders", "orders"),
#     ("find orders worth less than 100", "orders"),
#     ("show distinct order statuses", "orders"),
# ]
#
# role_encoder = SchemaRoleEncoder(tokenizer=tokenizer)
# schema_token_ids, schema_role_ids = role_encoder.encode_schema(schema)
#
# candidate_tables = list(schema.keys())
#
# correct = 0
#
# for question, expected_table in table_test_cases:
#     question_prompt = prompt_builder.build_question_prompt(question)
#     question_start = question_prompt.find("Question:")
#     question_ids = tokenizer.encode_ids(question_prompt[question_start:])
#
#     input_ids = schema_token_ids + question_ids
#     role_ids = schema_role_ids + [SchemaRoleEncoder.NORMAL] * len(question_ids)
#
#     scores = []
#
#     with torch.no_grad():
#         for table in candidate_tables:
#             table_token_ids = tokenizer.encode_ids(table)
#
#             score = model.score_table(
#                 input_ids=input_ids,
#                 schema_role_ids=role_ids,
#                 table_token_ids=table_token_ids,
#             )
#
#             scores.append((table, score.item()))
#
#     scores.sort(key=lambda item: item[1], reverse=True)
#
#     predicted_table = scores[0][0]
#
#     if predicted_table == expected_table:
#         correct += 1
#         result = "✓"
#     else:
#         result = "✗"
#
#     print(f"\n{result} Question : {question}")
#     print(f"  Expected : {expected_table}")
#     print(f"  Predicted: {predicted_table}")
#
# accuracy = correct / len(table_test_cases)
#
# print()
# print(f"Correct             : {correct}/{len(table_test_cases)}")
# print(f"Table Head Accuracy : {accuracy:.4f}")
# print("==================================\n")
#
# print("\n===== COLUMN HEAD DIAGNOSTIC =====")

diagnostic_questions = [
    "show employees with salary greater than 60000",
    "show employees with age greater than 30",
]

role_encoder = SchemaRoleEncoder(tokenizer=tokenizer)
schema_token_ids, schema_role_ids = role_encoder.encode_schema(schema)

employee_columns = schema["employees"]

for question in diagnostic_questions:
    question_prompt = prompt_builder.build_question_prompt(question)
    question_start = question_prompt.find("Question:")

    question_ids = tokenizer.encode_ids(
        question_prompt[question_start:]
    )

    input_ids = schema_token_ids + question_ids
    role_ids = (
        schema_role_ids
        + [SchemaRoleEncoder.NORMAL] * len(question_ids)
    )

    scores = []

    with torch.no_grad():
        for column in employee_columns:
            column_token_ids = tokenizer.encode_ids(column)

            score = model.score_column(
                input_ids=input_ids,
                schema_role_ids=role_ids,
                column_token_ids=column_token_ids,
            )

            scores.append((column, score.item()))

    scores.sort(key=lambda item: item[1], reverse=True)

    print("\nQuestion:", question)

    for column, score in scores:
        print(f"  {column:20} {score:.4f}")

print("\n==================================\n")

print("\n===== COLUMN HEAD ACCURACY TEST =====")

column_test_cases = [
    ("show employees with salary greater than 60000", "salary"),
    ("find employees earning more than 60000", "salary"),
    ("employees whose salary exceeds 50000", "salary"),

    ("show employees with age greater than 30", "age"),
    ("find employees older than 30", "age"),
    ("employees younger than 40", "age"),

    ("show products with price greater than 50", "price"),
    ("find products cheaper than 100", "price"),

    ("show orders above 100", "total_amount"),
    ("find orders worth less than 100", "total_amount"),
]

role_encoder = SchemaRoleEncoder(tokenizer=tokenizer)
schema_token_ids, schema_role_ids = role_encoder.encode_schema(schema)

candidate_columns = []

for columns in schema.values():
    for column in columns:
        if column not in candidate_columns:
            candidate_columns.append(column)

correct = 0

for question, expected_column in column_test_cases:
    question_prompt = prompt_builder.build_question_prompt(question)
    question_start = question_prompt.find("Question:")

    question_ids = tokenizer.encode_ids(question_prompt[question_start:])

    input_ids = schema_token_ids + question_ids
    role_ids = schema_role_ids + [SchemaRoleEncoder.NORMAL] * len(question_ids)

    scores = []

    with torch.no_grad():
        for column in candidate_columns:
            column_token_ids = tokenizer.encode_ids(column)

            score = model.score_column(
                input_ids=input_ids,
                schema_role_ids=role_ids,
                column_token_ids=column_token_ids,
            )

            scores.append((column, score.item()))

    scores.sort(key=lambda item: item[1], reverse=True)

    predicted_column = scores[0][0]

    if predicted_column == expected_column:
        correct += 1
        result = "✓"
    else:
        result = "✗"

    print(f"\n{result} Question : {question}")
    print(f"  Expected : {expected_column}")
    print(f"  Predicted: {predicted_column}")

accuracy = correct / len(column_test_cases)

print()
print(f"Correct              : {correct}/{len(column_test_cases)}")
print(f"Column Head Accuracy : {accuracy:.4f}")
print("=====================================\n")

# print("\n===== TINY WIKISQL OVERFIT TEST =====")
#
# train_tiny_gpt(
#     model=model,
#     training_dataset=training_dataset,
#     epochs=100,
#     learning_rate=0.001,
#     batch_size=1
# )
print("\n===== ORDER BY DIRECTION DISTRIBUTION =====")

asc_examples = []
desc_examples = []

for example in corpus:
    sql = example["sql"].upper()

    if "ORDER BY" not in sql:
        continue

    if " ASC" in sql:
        asc_examples.append(example)

    if " DESC" in sql:
        desc_examples.append(example)

print("ASC :", len(asc_examples))
print("DESC:", len(desc_examples))

print("\nASC examples:")
for example in asc_examples[:10]:
    print("Question:", example["prompt"])
    print("SQL     :", example["sql"])

print("\nDESC examples:")
for example in desc_examples[:10]:
    print("Question:", example["prompt"])
    print("SQL     :", example["sql"])

print("===========================================\n")

print("\n===== ORDER DIRECTION DIAGNOSTIC =====")

direction_tests = [
    ("show employees ordered by salary lowest first",
     "SELECT * FROM employees ORDER BY salary"),

    ("show employees ordered by salary highest first",
     "SELECT * FROM employees ORDER BY salary"),

    ("show products ordered by price lowest first",
     "SELECT * FROM products ORDER BY price"),

    ("show products ordered by price highest first",
     "SELECT * FROM products ORDER BY price"),
]

for question, sql_prefix in direction_tests:
    role_encoder = SchemaRoleEncoder(tokenizer=tokenizer)
    schema_ids, schema_roles = role_encoder.encode_schema(schema)

    question_prompt = prompt_builder.build_question_prompt(question)
    question_start = question_prompt.find("Question:")
    question_ids = tokenizer.encode_ids(question_prompt[question_start:])

    prompt_ids = schema_ids + question_ids
    prompt_roles = schema_roles + [SchemaRoleEncoder.NORMAL] * len(question_ids)

    prefix_ids = tokenizer.encode_ids(sql_prefix + " ")

    input_ids = prompt_ids + prefix_ids
    role_ids = prompt_roles + [SchemaRoleEncoder.NORMAL] * len(prefix_ids)

    with torch.no_grad():
        logits = model(input_ids, role_ids)
        log_probs = torch.log_softmax(logits[0, -1], dim=-1)

    asc_ids = tokenizer.encode_ids("ASC")
    desc_ids = tokenizer.encode_ids("DESC")

    print("\nQuestion:", question)
    print("ASC :", log_probs[asc_ids[0]].item())
    print("DESC:", log_probs[desc_ids[0]].item())
    print(
        "MODEL PREFERS:",
        "ASC" if log_probs[asc_ids[0]] > log_probs[desc_ids[0]] else "DESC"
    )

print("\n======================================\n")

def score_full_sql(question: str, sql: str) -> tuple[float, float]:
    role_encoder = SchemaRoleEncoder(tokenizer=tokenizer)
    schema_token_ids, schema_role_ids = role_encoder.encode_schema(schema)

    question_prompt = prompt_builder.build_question_prompt(question)
    question_start = question_prompt.find("Question:")
    question_ids = tokenizer.encode_ids(question_prompt[question_start:])
    sql_ids = tokenizer.encode_ids(sql)

    current_ids = schema_token_ids + question_ids
    current_role_ids = schema_role_ids + [SchemaRoleEncoder.NORMAL] * len(question_ids)

    total_score = 0.0

    model.eval()

    with torch.no_grad():
        for token_id in sql_ids:
            input_tensor = torch.tensor(current_ids, dtype=torch.long)
            role_tensor = torch.tensor(current_role_ids, dtype=torch.long)

            logits = model(input_tensor, role_tensor)
            log_probs = torch.log_softmax(logits[0, -1], dim=-1)

            token_score = log_probs[token_id].item()
            total_score += token_score

            print(f"{repr(tokenizer.decode_ids([token_id])):18} {token_score:.4f}")

            current_ids.append(token_id)
            current_role_ids.append(SchemaRoleEncoder.NORMAL)

    average_score = total_score / len(sql_ids)

    return total_score, average_score

def score_column_candidate(question: str, sql_prefix: str, column: str) -> float:
    role_encoder = SchemaRoleEncoder(tokenizer=tokenizer)
    schema_ids, schema_roles = role_encoder.encode_schema(schema)

    question_prompt = prompt_builder.build_question_prompt(question)
    question_start = question_prompt.find("Question:")
    question_ids = tokenizer.encode_ids(question_prompt[question_start:])

    prefix_ids = tokenizer.encode_ids(sql_prefix)
    column_ids = tokenizer.encode_ids(column)

    working_ids = schema_ids + question_ids + prefix_ids
    working_roles = (
        schema_roles
        + [SchemaRoleEncoder.NORMAL] * len(question_ids)
        + [SchemaRoleEncoder.NORMAL] * len(prefix_ids)
    )

    total_score = 0.0

    with torch.no_grad():
        for token_id in column_ids:
            logits = model(working_ids, working_roles)
            log_probs = torch.log_softmax(logits[0, -1], dim=-1)

            total_score += log_probs[token_id].item()

            working_ids.append(token_id)
            working_roles.append(SchemaRoleEncoder.NORMAL)

    return total_score / len(column_ids)

print("\n===== ORDERS COLUMN DIAGNOSTIC =====")

orders_column_tests = [
    (
        "show orders where quantity is greater than 2",
        "SELECT * FROM orders WHERE ",
        "quantity",
    ),
    (
        "show orders where total amount is greater than 100",
        "SELECT * FROM orders WHERE ",
        "total_amount",
    ),
    (
        "what is the average order quantity",
        "SELECT AVG(",
        "quantity",
    ),
    (
        "what is the average order amount",
        "SELECT AVG(",
        "total_amount",
    ),
    (
        "show orders ordered by quantity lowest first",
        "SELECT * FROM orders ORDER BY ",
        "quantity",
    ),
    (
        "show orders ordered by total amount lowest first",
        "SELECT * FROM orders ORDER BY ",
        "total_amount",
    ),
]

for question, sql_prefix, expected_column in orders_column_tests:
    quantity_score = score_column_candidate(
        question,
        sql_prefix,
        "quantity"
    )

    amount_score = score_column_candidate(
        question,
        sql_prefix,
        "total_amount"
    )

    predicted = (
        "quantity"
        if quantity_score > amount_score
        else "total_amount"
    )

    print("\nQuestion :", question)
    print("Expected :", expected_column)
    print(f"quantity     : {quantity_score:.4f}")
    print(f"total_amount : {amount_score:.4f}")
    print("MODEL PREFERS:", predicted)

print("\n====================================\n")

def score_operator_candidate(question: str, sql_prefix: str, operator_candidate: str) -> float:
    role_encoder = SchemaRoleEncoder(tokenizer=tokenizer)

    schema_token_ids, schema_role_ids = role_encoder.encode_schema(schema)

    question_prompt = prompt_builder.build_question_prompt(question)
    question_start = question_prompt.find("Question:")

    if question_start == -1:
        raise ValueError("Prompt must contain 'Question:'.")

    question_ids = tokenizer.encode_ids(question_prompt[question_start:])
    prefix_ids = tokenizer.encode_ids(sql_prefix)
    candidate_ids = tokenizer.encode_ids(operator_candidate)

    current_ids = schema_token_ids + question_ids + prefix_ids

    current_role_ids = (
        schema_role_ids
        + [SchemaRoleEncoder.NORMAL] * len(question_ids)
        + [SchemaRoleEncoder.NORMAL] * len(prefix_ids)
    )

    total_log_probability = 0.0

    model.eval()

    with torch.no_grad():
        for token_id in candidate_ids:
            input_tensor = torch.tensor(current_ids, dtype=torch.long)
            role_tensor = torch.tensor(current_role_ids, dtype=torch.long)

            logits = model(input_tensor, role_tensor)
            next_token_logits = logits[0, -1]

            log_probabilities = torch.log_softmax(next_token_logits, dim=-1)
            total_log_probability += log_probabilities[token_id].item()

            current_ids.append(token_id)
            current_role_ids.append(SchemaRoleEncoder.NORMAL)

    return total_log_probability

def compare_operators(question: str,sql_prefix: str) -> None:

    greater_score = score_operator_candidate(
        question=question,
        sql_prefix=sql_prefix,
        operator_candidate=" > ",
    )

    less_score = score_operator_candidate(
        question=question,
        sql_prefix=sql_prefix,
        operator_candidate=" < ",
    )

    print("\nQuestion:", question)
    print("SQL prefix:", sql_prefix)
    print("> score:", greater_score)
    print("< score:", less_score)

    if greater_score > less_score:
        print("MODEL PREFERS: >")
    else:
        print("MODEL PREFERS: <")

def compare_in_operator(question: str, sql_prefix: str) -> None:
    candidates = [" = ", " > ", " < ", " IN "]

    print("\nQuestion:", question)
    print("SQL prefix:", sql_prefix)

    scores = {}

    for candidate in candidates:
        score = score_operator_candidate(
            question=question,
            sql_prefix=sql_prefix,
            operator_candidate=candidate,
        )

        scores[candidate.strip()] = score
        print(f"{candidate.strip():2}: {score}")

    print("MODEL PREFERS:", max(scores, key=scores.get))

print("\n===== IN OPERATOR GENERALIZATION DIAGNOSTIC =====")

compare_in_operator(
    question="employees who are either 26 or 31 years old",
    sql_prefix="SELECT * FROM employees WHERE age",
)

compare_in_operator(
    question="products costing either 45 or 120",
    sql_prefix="SELECT * FROM products WHERE price",
)

compare_in_operator(
    question="orders containing either 1 or 3 items",
    sql_prefix="SELECT * FROM orders WHERE quantity",
)

print("=================================================\n")

print("\n===== EQUALITY OPERATOR DIAGNOSTIC =====")

compare_in_operator(
    question="employees aged exactly 30",
    sql_prefix="SELECT * FROM employees WHERE age",
)

compare_in_operator(
    question="products costing exactly 50",
    sql_prefix="SELECT * FROM products WHERE price",
)

compare_in_operator(
    question="orders with exactly 2 items",
    sql_prefix="SELECT * FROM orders WHERE quantity",
)

print("=========================================\n")


print("\n===== IN WHERE COLUMN DIAGNOSTIC =====")

question = "employees who are either 26 or 31 years old"
sql_prefix = "SELECT * FROM employees WHERE "

for column in schema["employees"]:
    score = score_column_candidate(
        question=question,
        sql_prefix=sql_prefix,
        column=column,
    )
    print(f"{column:20} {score:.4f}")

print("======================================\n")


print("\n===== OPERATOR DIAGNOSTIC: TRAINING-LIKE =====")

compare_operators(
    question="show employees older than 30",
    sql_prefix="SELECT * FROM employees WHERE age",
)

compare_operators(
    question="show employees younger than 30",
    sql_prefix="SELECT * FROM employees WHERE age",
)

compare_operators(
    question="show products priced above 50",
    sql_prefix="SELECT * FROM products WHERE price",
)

compare_operators(
    question="show products cheaper than 50",
    sql_prefix="SELECT * FROM products WHERE price",
)

print("\n===== OPERATOR DIAGNOSTIC: UNSEEN =====")

compare_operators(
    question="find employees under 30 years old",
    sql_prefix="SELECT * FROM employees WHERE age",
)

compare_operators(
    question="which products cost under 50",
    sql_prefix="SELECT * FROM products WHERE price",
)

compare_operators(
    question="find orders worth less than 100",
    sql_prefix="SELECT * FROM orders WHERE total_amount",
)


# question = "show all products"
# prompt = prompt_builder.build_question_prompt(question)
#
# prompt_ids = tokenizer.encode_ids(prompt)
# sql_prefix_ids = tokenizer.encode_ids("SELECT * FROM ")
#
# input_ids = prompt_ids + sql_prefix_ids
# role_ids = [SchemaRoleEncoder.NORMAL] * len(input_ids)
#
# with torch.no_grad():
#     logits = model(
#         torch.tensor(input_ids, dtype=torch.long),
#         torch.tensor(role_ids, dtype=torch.long)
#     )
#
# last_logits = logits[0, -1]

# for table in ["employees", "products", "orders"]:
#     table_id = tokenizer.encode_ids(table)[0]
#     print(table, last_logits[table_id].item())
print("\n===== TRAINING ACCURACY =====")

correct_tokens = 0
total_tokens = 0

correct_examples = 0

custom_correct = 0
custom_total = len(corpus)

wikisql_correct = 0
wikisql_total = len(wikisql_corpus)

for index, example in enumerate(training_dataset):
    input_ids = torch.tensor(example["input_ids"], dtype=torch.long)
    target_ids = torch.tensor(example["target_ids"], dtype=torch.long)
    role_ids = torch.tensor(example["schema_role_ids"], dtype=torch.long)
    prompt_length = example["prompt_length"]

    with torch.no_grad():
        logits = model(input_ids, role_ids)

    sql_logits = logits[0, prompt_length - 1:]
    predicted_ids = sql_logits.argmax(dim=-1)

    correct_tokens += (predicted_ids == target_ids).sum().item()
    total_tokens += len(target_ids)

    is_exact = torch.equal(predicted_ids, target_ids)

    if is_exact:
        correct_examples += 1

        if index < custom_total:
            custom_correct += 1
        else:
            wikisql_correct += 1

print("Token accuracy          :", correct_tokens / total_tokens)
print("Overall exact accuracy  :",correct_examples / len(training_dataset),f"({correct_examples}/{len(training_dataset)})")
print("Custom exact accuracy   :",custom_correct / custom_total,f"({custom_correct}/{custom_total})")
print("WikiSQL exact accuracy  :",wikisql_correct / wikisql_total,f"({wikisql_correct}/{wikisql_total})")
print("=============================\n")


print("\n===== OPERATOR HEAD TEST =====")

operator_names = {
    0: "NONE",
    1: ">",
    2: "<",
    3: "=",
    4: "IN",
}

operator_tests = [
    "show employees older than 30",
    "show employees younger than 30",
    "show products priced above 50",
    "show products cheaper than 50",
    "find employees under 30 years old",
    "which products cost under 50",
    "find orders worth less than 100",
    "show employees where age equals 30",
    "show products where price equals 50",
    "show orders where quantity equals 2",
    "employees aged exactly 30",
    "products costing exactly 50",
    "orders with exactly 2 items",
]

# for question in operator_tests:
#     prompt = prompt_builder.build_question_prompt(question)
#     # prompt = prompt[prompt.find("Question:"):]
#
#     input_ids = tokenizer.encode_ids(prompt)
#
#     with torch.no_grad():
#         predicted = model.predict_operator(input_ids).argmax(dim=-1).item()
#
#     print(question, "->", operator_names[predicted])


role_encoder = SchemaRoleEncoder(tokenizer=tokenizer)
schema_token_ids, schema_role_ids = role_encoder.encode_schema(schema)

for question in operator_tests:
    question_prompt = prompt_builder.build_question_prompt(question)
    question_start = question_prompt.find("Question:")
    question_ids = tokenizer.encode_ids(question_prompt[question_start:])
    input_ids = schema_token_ids + question_ids
    role_ids = schema_role_ids + [SchemaRoleEncoder.NORMAL] * len(question_ids)
    with torch.no_grad():
        predicted = model.predict_operator(input_ids, role_ids).argmax(dim=-1).item()
    print(question, "->", operator_names[predicted])
print("================================\n")
# exit()

# print("\n===== SHOW PREFIX DIAGNOSTIC =====")
#
# question = "show employees who are either 26 or 31 years old"
#
# wrong_sql = "SELECT * FROM employees WHERE salary < 31;"
# correct_sql = "SELECT * FROM employees WHERE age IN (26, 31);"
#
# wrong_total, wrong_avg = score_full_sql(question, wrong_sql)
# correct_total, correct_avg = score_full_sql(question, correct_sql)
#
# print("Wrong average  :", wrong_avg)
# print("Correct average:", correct_avg)
# print("MODEL PREFERS:", "CORRECT" if correct_avg > wrong_avg else "WRONG")
#
# exit()

# print("\n===== UNSEEN WORDING + FAMILIAR VALUES =====")
#
# question = "employees who are either 25 or 30 years old"
#
# wrong_sql = "SELECT * FROM employees WHERE age < 30;"
# correct_sql = "SELECT * FROM employees WHERE age IN (25, 30);"
#
# wrong_total, wrong_avg = score_full_sql(question, wrong_sql)
# correct_total, correct_avg = score_full_sql(question, correct_sql)
#
# print("Wrong average  :", wrong_avg)
# print("Correct average:", correct_avg)
# print("MODEL PREFERS:", "CORRECT" if correct_avg > wrong_avg else "WRONG")
#
# exit()
#
# question = "show employees aged 26 or 31"
#
# wrong_sql = "SELECT * FROM employees WHERE age < 31;"
# correct_sql = "SELECT * FROM employees WHERE age IN (26, 31);"
#
# print("\n===== FAMILIAR WORDING + UNSEEN VALUES =====")
#
# wrong_total, wrong_avg = score_full_sql(question, wrong_sql)
# correct_total, correct_avg = score_full_sql(question, correct_sql)
#
# print("Wrong average  :", wrong_avg)
# print("Correct average:", correct_avg)
# print("MODEL PREFERS:", "CORRECT" if correct_avg > wrong_avg else "WRONG")
#
# exit()

# print("\n===== TRAINING-LIKE IN FULL SCORE =====")
#
# question = "show employees aged 25 or 30"
#
# wrong_sql = "SELECT * FROM employees WHERE age < 30;"
# correct_sql = "SELECT * FROM employees WHERE age IN (25, 30);"
#
# print("\n===== WRONG SQL =====")
# wrong_total, wrong_avg = score_full_sql(question, wrong_sql)
# print("TOTAL:", wrong_total)
# print("AVG  :", wrong_avg)
#
# print("\n===== CORRECT SQL =====")
# correct_total, correct_avg = score_full_sql(question, correct_sql)
# print("TOTAL:", correct_total)
# print("AVG  :", correct_avg)
#
# print("\n===== COMPARISON =====")
# print("Wrong average  :", wrong_avg)
# print("Correct average:", correct_avg)
# print("MODEL PREFERS:", "CORRECT" if correct_avg > wrong_avg else "WRONG")
#
# exit()

# print("\n===== FULL SQL SCORE DIAGNOSTIC =====")
#
# question = "employees who are either 26 or 31 years old"
#
# wrong_sql = "SELECT * FROM employees WHERE salary < 31;"
# correct_sql = "SELECT * FROM employees WHERE age IN (26, 31);"
#
# print("\n===== WRONG SQL =====")
# wrong_total, wrong_avg = score_full_sql(question, wrong_sql)
# print("TOTAL:", wrong_total)
# print("AVG  :", wrong_avg)
#
# print("\n===== CORRECT SQL =====")
# correct_total, correct_avg = score_full_sql(question, correct_sql)
# print("TOTAL:", correct_total)
# print("AVG  :", correct_avg)
#
# print("\n===== COMPARISON =====")
# print("Wrong average  :", wrong_avg)
# print("Correct average:", correct_avg)
#
# if correct_avg > wrong_avg:
#     print("MODEL PREFERS: CORRECT SQL")
# else:
#     print("MODEL PREFERS: WRONG SQL")
#
# print("=====================================")
# exit()


# ============================================================
# TEXT-to-SQL PIPELINE
# ============================================================

pipeline = TextToSQLPipeline(
    model=model,
    tokenizer=tokenizer,
    schema=schema,
    connection=connection,
    beam_width=5,
)

print("\n===== V4 IN + EQUALITY TEST =====")

tests = [
    # IN
    ("employees who are either 26 or 31 years old",
     "SELECT * FROM employees WHERE age IN (26, 31);"),

    ("products costing either 45 or 120",
     "SELECT * FROM products WHERE price IN (45, 120);"),

    ("orders containing either 1 or 3 items",
     "SELECT * FROM orders WHERE quantity IN (1, 3);"),

    # EQUAL
    ("employees aged exactly 30",
     "SELECT * FROM employees WHERE age = 30;"),

    ("products costing exactly 50",
     "SELECT * FROM products WHERE price = 50;"),

    ("orders with exactly 2 items",
     "SELECT * FROM orders WHERE quantity = 2;"),
]

correct = 0

for question, expected in tests:
    generated = pipeline.generate(question)
    passed = generated == expected

    if passed:
        correct += 1

    print("\nQuestion :", question)
    print("Predicted:", generated)
    print("Expected :", expected)
    print("PASS     :", passed)

print(f"\nRESULT: {correct}/{len(tests)}")
print("================================")
# exit()

# print("\n===== IN SCORING REGRESSION TEST =====")
#
# tests = [
#     ("employees who are either 26 or 31 years old",
#      "SELECT * FROM employees WHERE age IN (26, 31);"),
#
#     ("products costing either 45 or 120",
#      "SELECT * FROM products WHERE price IN (45, 120);"),
#
#     ("orders containing either 1 or 3 items",
#      "SELECT * FROM orders WHERE quantity IN (1, 3);"),
#
#     ("employees aged exactly 30",
#      "SELECT * FROM employees WHERE age = 30;"),
#
#     ("products costing exactly 50",
#      "SELECT * FROM products WHERE price = 50;"),
#
#     ("orders with exactly 2 items",
#      "SELECT * FROM orders WHERE quantity = 2;"),
# ]
#
# for question, expected in tests:
#     generated = pipeline.generate(question)
#
#     print("\nQuestion :", question)
#     print("Predicted:", generated)
#     print("Expected :", expected)
#     print("PASS     :", generated == expected)
#
# print("======================================")
# exit()

print("\n===== FORCED IN SCORING TEST =====")

question = "employees who are either 26 or 31 years old"

print("Question:", question)
print("SQL:", pipeline.generate(question))

print("================================")

print("\n===== HELD-OUT IN TEST =====")

tests = [
    (
        "employees who are either 26 or 31 years old",
        "SELECT * FROM employees WHERE age IN (26, 31);"
    ),
    (
        "products costing either 45 or 120",
        "SELECT * FROM products WHERE price IN (45, 120);"
    ),
    (
        "orders containing either 1 or 3 items",
        "SELECT * FROM orders WHERE quantity IN (1, 3);"
    ),
]

for question, expected in tests:
    generated = pipeline.generate(question)

    print("\nQuestion :", question)
    print("Predicted:", generated)
    print("Expected :", expected)
    print("PASS     :", generated == expected)

# exit()

print("\n===== V3 FULL SQL SCORE DIAGNOSTIC =====")

question = "employees who are either 26 or 31 years old"

wrong_sql = "SELECT * FROM employees WHERE salary < 26;"
correct_sql = "SELECT * FROM employees WHERE age IN (26, 31);"

print("\n===== WRONG SQL =====")
wrong_total, wrong_avg = score_full_sql(question, wrong_sql)
print("TOTAL:", wrong_total)
print("AVG  :", wrong_avg)

print("\n===== CORRECT SQL =====")
correct_total, correct_avg = score_full_sql(question, correct_sql)
print("TOTAL:", correct_total)
print("AVG  :", correct_avg)

print("\n===== COMPARISON =====")
print("Wrong average  :", wrong_avg)
print("Correct average:", correct_avg)
print("MODEL PREFERS:", "CORRECT" if correct_avg > wrong_avg else "WRONG")

# exit()

# print("\n===== COLUMN HEAD ABLATION TEST =====")
#
# question = "employees who are either 26 or 31 years old"
#
# print("Question:", question)
# print("SQL:", pipeline.generate(question))
#
# print("=====================================")
# exit()
print("\n===== COLUMN HEAD ABLATION TEST =====")

question = "employees who are either 26 or 31 years old"

print("Question:", question)
print("SQL:", pipeline.generate(question))

print("=====================================")
# exit()
# print("\n===== IN VALUE DEBUG =====")
#
# debug_questions = [
#     # "employees who are either 26 or 31 years old",
#     # "products costing either 45 or 120",
#     "orders containing either 1 or 3 items",
# ]
#
# for question in debug_questions:
#     print("\nQuestion:", question)
#     print("SQL:", pipeline.generate(question))
#
# print("==========================")
# exit()

print("\n===== TABLE CHOICE DIAGNOSTIC =====")

table_tests = [
    ("show all employees", "employees"),
    ("show all products", "products"),
    ("show all orders", "orders"),
]

for question, expected_table in table_tests:
    role_encoder = SchemaRoleEncoder(tokenizer=tokenizer)
    schema_token_ids, schema_role_ids = role_encoder.encode_schema(schema)

    question_prompt = prompt_builder.build_question_prompt(question)
    question_start = question_prompt.find("Question:")

    if question_start == -1:
        raise ValueError("Prompt must contain 'Question:'.")

    question_ids = tokenizer.encode_ids(question_prompt[question_start:])

    prompt_ids = schema_token_ids + question_ids
    prompt_role_ids = schema_role_ids + [SchemaRoleEncoder.NORMAL] * len(question_ids)

    # Force decoding up to the point immediately before TABLE
    sql_prefix = "SELECT * FROM "
    prefix_ids = tokenizer.encode_ids(sql_prefix)

    input_ids = prompt_ids + prefix_ids
    role_ids = prompt_role_ids + [SchemaRoleEncoder.NORMAL] * len(prefix_ids)

    table_scores = {}

    with torch.no_grad():
        for table in schema.keys():
            table_ids = tokenizer.encode_ids(table)

            working_ids = input_ids.copy()
            working_roles = role_ids.copy()
            total_score = 0.0

            for token_id in table_ids:
                logits = model(working_ids, working_roles)
                last_logits = logits[0, -1]

                log_probabilities = torch.log_softmax(last_logits, dim=-1)
                total_score += log_probabilities[token_id].item()

                working_ids.append(token_id)
                working_roles.append(SchemaRoleEncoder.NORMAL)

            # Normalize so longer BPE tokenizations aren't automatically penalized
            table_scores[table] = total_score / len(table_ids)

    print(f"\nQuestion: {question}")
    print(f"Expected: {expected_table}")

    for table, score in sorted(table_scores.items(), key=lambda item: item[1], reverse=True):
        print(f"  {table:12} {score:.4f}")

print("\n===================================\n")

print("\n===== BASIC SQL CAPABILITY TEST =====")

capability_tests = {
    "SELECT": [
        ("show all employees", "SELECT * FROM employees;"),
        ("show all products", "SELECT * FROM products;"),
        ("list all orders", "SELECT * FROM orders;"),
    ],

    "EQUAL": [
        ("show employees where age equals 30", "SELECT * FROM employees WHERE age = 30;"),
        ("show products where price equals 50", "SELECT * FROM products WHERE price = 50;"),
        ("show orders where quantity equals 2", "SELECT * FROM orders WHERE quantity = 2;"),
    ],

    "GREATER THAN": [
        ("show employees where age is greater than 30", "SELECT * FROM employees WHERE age > 30;"),
        ("show products where price is greater than 50", "SELECT * FROM products WHERE price > 50;"),
        ("show orders where quantity is greater than 2", "SELECT * FROM orders WHERE quantity > 2;"),
    ],

    "LESS THAN": [
        ("show employees where age is less than 30", "SELECT * FROM employees WHERE age < 30;"),
        ("show products where price is less than 50", "SELECT * FROM products WHERE price < 50;"),
        ("show orders where quantity is less than 2", "SELECT * FROM orders WHERE quantity < 2;"),
    ],
    "IN": [
        ("show employees aged 25 or 30",
         "SELECT * FROM employees WHERE age IN (25, 30);"),

        ("show products priced 50 or 100",
         "SELECT * FROM products WHERE price IN (50, 100);"),

        ("show orders with quantity 1 or 2",
         "SELECT * FROM orders WHERE quantity IN (1, 2);"),
    ],

    "COUNT": [
        ("how many employees are there", "SELECT COUNT(*) FROM employees;"),
        ("count all products", "SELECT COUNT(*) FROM products;"),
        ("how many orders are there", "SELECT COUNT(*) FROM orders;"),
    ],

    "AVG": [
        ("what is the average employee salary", "SELECT AVG(salary) FROM employees;"),
        ("what is the average product price", "SELECT AVG(price) FROM products;"),
        ("what is the average order quantity", "SELECT AVG(quantity) FROM orders;"),
    ],

    "DISTINCT": [
        ("show unique employee departments", "SELECT DISTINCT department FROM employees;"),
        ("show unique product categories", "SELECT DISTINCT category FROM products;"),
        ("show unique order statuses", "SELECT DISTINCT status FROM orders;"),
    ],

    "ORDER BY ASC": [
        ("show employees ordered by salary lowest first", "SELECT * FROM employees ORDER BY salary ASC;"),
        ("show products ordered by price lowest first", "SELECT * FROM products ORDER BY price ASC;"),
        ("show orders ordered by quantity lowest first", "SELECT * FROM orders ORDER BY quantity ASC;"),
    ],

    "ORDER BY DESC": [
        ("show employees ordered by salary highest first", "SELECT * FROM employees ORDER BY salary DESC;"),
        ("show products ordered by price highest first", "SELECT * FROM products ORDER BY price DESC;"),
        ("show orders ordered by quantity highest first", "SELECT * FROM orders ORDER BY quantity DESC;"),
    ],

    "LIMIT": [
        ("show first 5 employees", "SELECT * FROM employees LIMIT 5;"),
        ("show first 5 products", "SELECT * FROM products LIMIT 5;"),
        ("show first 5 orders", "SELECT * FROM orders LIMIT 5;"),
    ],
}

capability_results = {}

for capability, tests in capability_tests.items():
    correct = 0

    print(f"\n--- {capability} ---")

    for question, expected_sql in tests:
        try:
            generated_sql = pipeline.generate(question)
            passed = generated_sql.strip() == expected_sql.strip()

            if passed:
                correct += 1

            print("✓" if passed else "✗", question)

            if not passed:
                print("  Expected :", expected_sql)
                print("  Generated:", generated_sql)

        except Exception as error:
            print("✗", question)
            print("  Error:", error)

    capability_results[capability] = (correct, len(tests))


print("\n===== CAPABILITY SUMMARY =====")

total_correct = 0
total_tests = 0

for capability, (correct, total) in capability_results.items():
    total_correct += correct
    total_tests += total

    print(f"{capability:15} {correct}/{total}")

print("------------------------------")
print(f"Overall         {total_correct}/{total_tests}")
print(f"Accuracy        {total_correct / total_tests:.2%}")
print("==============================\n")

print("\n===== VALUE VS COLUMN DIAGNOSTIC =====")

diagnostic_questions = [
    "show employees with salary greater than 30",
    "show employees with salary greater than 50",
    "show employees with salary greater than 100",
    "show employees with salary greater than 60000",

    "show employees with age greater than 30",
    "show employees with age greater than 50",
    "show employees with age greater than 100",
    "show employees with age greater than 60000",
]

role_encoder = SchemaRoleEncoder(tokenizer=tokenizer)
schema_token_ids, schema_role_ids = role_encoder.encode_schema(schema)

employee_columns = schema["employees"]

model.eval()

for question in diagnostic_questions:
    question_prompt = prompt_builder.build_question_prompt(question)
    question_start = question_prompt.find("Question:")
    question_ids = tokenizer.encode_ids(question_prompt[question_start:])

    prefix_ids = tokenizer.encode_ids("SELECT * FROM employees WHERE ")

    input_ids = schema_token_ids + question_ids + prefix_ids
    role_ids = (
        schema_role_ids
        + [SchemaRoleEncoder.NORMAL] * len(question_ids)
        + [SchemaRoleEncoder.NORMAL] * len(prefix_ids)
    )

    scores = []

    with torch.no_grad():
        for column in employee_columns:
            candidate_ids = tokenizer.encode_ids(column)

            working_ids = input_ids.copy()
            working_roles = role_ids.copy()
            total_log_probability = 0.0

            for token_id in candidate_ids:
                logits = model(
                    torch.tensor(working_ids, dtype=torch.long),
                    torch.tensor(working_roles, dtype=torch.long),
                )

                next_logits = logits[0, -1]
                log_probs = torch.log_softmax(next_logits, dim=-1)

                total_log_probability += log_probs[token_id].item()

                working_ids.append(token_id)
                working_roles.append(SchemaRoleEncoder.NORMAL)

            normalized_score = total_log_probability / len(candidate_ids)
            scores.append((column, normalized_score))

    scores.sort(key=lambda item: item[1], reverse=True)

    print("\nQuestion:", question)

    for column, score in scores[:3]:
        print(f"  {column:20} {score:.4f}")

print("\n======================================")

print("\n===== WHERE COLUMN RANKING =====")

question = "show employees with salary greater than 60000"

role_encoder = SchemaRoleEncoder(tokenizer=tokenizer)
schema_token_ids, schema_role_ids = role_encoder.encode_schema(schema)

question_prompt = prompt_builder.build_question_prompt(question)
question_start = question_prompt.find("Question:")
question_ids = tokenizer.encode_ids(question_prompt[question_start:])

sql_prefix = "SELECT * FROM employees WHERE "
prefix_ids = tokenizer.encode_ids(sql_prefix)

input_ids = schema_token_ids + question_ids + prefix_ids
role_ids = (
    schema_role_ids
    + [SchemaRoleEncoder.NORMAL] * len(question_ids)
    + [SchemaRoleEncoder.NORMAL] * len(prefix_ids)
)

employee_columns = schema["employees"]

scores = []

model.eval()

with torch.no_grad():
    for column in employee_columns:
        candidate_ids = tokenizer.encode_ids(column)

        working_ids = input_ids.copy()
        working_roles = role_ids.copy()

        total_log_probability = 0.0

        for token_id in candidate_ids:
            logits = model(
                torch.tensor(working_ids, dtype=torch.long),
                torch.tensor(working_roles, dtype=torch.long),
            )

            next_logits = logits[0, -1]
            log_probs = torch.log_softmax(next_logits, dim=-1)

            total_log_probability += log_probs[token_id].item()

            working_ids.append(token_id)
            working_roles.append(SchemaRoleEncoder.NORMAL)

        normalized_score = total_log_probability / len(candidate_ids)

        scores.append(
            (column, total_log_probability, normalized_score)
        )

scores.sort(key=lambda item: item[2], reverse=True)

for column, raw_score, normalized_score in scores:
    print(
        f"{column:20} "
        f"raw={raw_score:.4f} "
        f"normalized={normalized_score:.4f}"
    )

print("===============================\n")


# ============================================================
# SINGLE QUESTION TEST
# ============================================================

question = "show all employees"

print("\n" + "=" * 60)
print("TEXT-to-SQL TEST")
print("=" * 60)

print("Question:", question)

generated_sql = pipeline.generate(question)

print("Generated SQL:", generated_sql)


# ============================================================
# EXECUTE GENERATED SQL
# ============================================================

try:
    cursor.execute(generated_sql)
    rows = cursor.fetchall()

    print("Execution: SUCCESS")
    print("Rows:")

    for row in rows:
        print(row)

except sqlite3.Error as error:
    print("Execution: FAILED")
    print("Error:", error)


# ============================================================
# SQL EVALUATION
# ============================================================

expected_sql = "SELECT * FROM employees;"

evaluator = SQLEvaluator(connection)

print("\nEvaluation:")
print("Exact match:", evaluator.exact_match(generated_sql, expected_sql))
print("Execution valid:", evaluator.execution_valid(generated_sql))
print("Execution match:", evaluator.execution_match(generated_sql, expected_sql))


# ============================================================
# MULTIPLE QUESTION TEST
# ============================================================

# test_cases = [
#     {
#         "question": "show all employees",
#         "expected_sql": "SELECT * FROM employees;",
#     },
#     {
#         "question": "how many products are there",
#         "expected_sql": "SELECT COUNT(*) FROM products;",
#     },
#     {
#         "question": "show all orders",
#         "expected_sql": "SELECT * FROM orders;",
#     },
#     {
#         "question": "show employees with salary greater than 60000",
#         "expected_sql": "SELECT * FROM employees WHERE salary > 60000;",
#     },
#     {
#         "question": "what is the average salary",
#         "expected_sql": "SELECT AVG(salary) FROM employees;",
#     },
#     {
#         "question": "show distinct departments",
#         "expected_sql": "SELECT DISTINCT department FROM employees;",
#     },
#     {
#         "question": "show the top 5 products",
#         "expected_sql": "SELECT * FROM products LIMIT 5;",
#     },
#     {
#         "question": "show employees ordered by salary from highest",
#         "expected_sql": "SELECT * FROM employees ORDER BY salary DESC;",
#     },
# ]

test_cases = [
    # SELECT ALL
    {"question": "show all employees",
     "expected_sql": "SELECT * FROM employees;"},
    {"question": "show all products",
     "expected_sql": "SELECT * FROM products;"},
    {"question": "show all orders",
     "expected_sql": "SELECT * FROM orders;"},

    # SALARY DIAGNOSTIC
    {"question": "show employees where salary is greater than 60000",
     "expected_sql": "SELECT * FROM employees WHERE salary > 60000;"},
    {"question": "find employees earning more than 60000",
     "expected_sql": "SELECT * FROM employees WHERE salary > 60000;"},

    # AGE DIAGNOSTIC
    {"question": "show employees with age greater than 30",
     "expected_sql": "SELECT * FROM employees WHERE age > 30;"},
    {"question": "show employees where age is greater than 30",
     "expected_sql": "SELECT * FROM employees WHERE age > 30;"},
    {"question": "find employees older than 30",
     "expected_sql": "SELECT * FROM employees WHERE age > 30;"},

    # COUNT
    {"question": "how many employees are there",
     "expected_sql": "SELECT COUNT(*) FROM employees;"},
    {"question": "how many products are there",
     "expected_sql": "SELECT COUNT(*) FROM products;"},
    {"question": "how many orders are there",
     "expected_sql": "SELECT COUNT(*) FROM orders;"},

    # SELECT COLUMN
    {"question": "show employee names",
     "expected_sql": "SELECT name FROM employees;"},
    {"question": "show employee salaries",
     "expected_sql": "SELECT salary FROM employees;"},
    {"question": "show product prices",
     "expected_sql": "SELECT price FROM products;"},
    {"question": "show order statuses",
     "expected_sql": "SELECT status FROM orders;"},

    # AVG
    {"question": "what is the average employee salary",
     "expected_sql": "SELECT AVG(salary) FROM employees;"},
    {"question": "what is the average product price",
     "expected_sql": "SELECT AVG(price) FROM products;"},
    {"question": "what is the average order amount",
     "expected_sql": "SELECT AVG(total_amount) FROM orders;"},

    # DISTINCT
    {"question": "show distinct employee departments",
     "expected_sql": "SELECT DISTINCT department FROM employees;"},
    {"question": "list unique departments",
     "expected_sql": "SELECT DISTINCT department FROM employees;"},

    {"question": "show distinct product categories",
     "expected_sql": "SELECT DISTINCT category FROM products;"},
    {"question": "list unique product categories",
     "expected_sql": "SELECT DISTINCT category FROM products;"},

    {"question": "show distinct order statuses",
     "expected_sql": "SELECT DISTINCT status FROM orders;"},
    {"question": "list unique order statuses",
     "expected_sql": "SELECT DISTINCT status FROM orders;"},
    #WHERE >
    {
        "question": "show employees with salary greater than 60000",
        "expected_sql": "SELECT * FROM employees WHERE salary > 60000;"
    },
    {
        "question": "show employees older than 30",
        "expected_sql": "SELECT * FROM employees WHERE age > 30;"
    },
    {
        "question": "show products priced above 50",
        "expected_sql": "SELECT * FROM products WHERE price > 50;"
    },
    {
        "question": "show orders above 100",
        "expected_sql": "SELECT * FROM orders WHERE total_amount > 100;"
    },
    {
        "question": "get products where price is greater than 50",
        "expected_sql": "SELECT * FROM products WHERE price > 50;"
    },
    #WHERE <
    {"question": "find employees under 30 years old",
     "expected_sql": "SELECT * FROM employees WHERE age < 30;"},

    {"question": "which products cost under 50",
     "expected_sql": "SELECT * FROM products WHERE price < 50;"},

    {"question": "find orders worth less than 100",
     "expected_sql": "SELECT * FROM orders WHERE total_amount < 100;"},
]


evaluation_examples = []

print("\n" + "=" * 60)
print("MULTIPLE QUESTION TEST")
print("=" * 60)

for test_case in test_cases:
    question = test_case["question"]
    expected_sql = test_case["expected_sql"]

    try:
        predicted_sql = pipeline.generate(question)
    except Exception as error:
        predicted_sql = ""
        print("\nQuestion:", question)
        print("Generation failed:", error)
        continue

    print("\nQuestion:", question)
    print("Predicted:", predicted_sql)
    print("Expected :", expected_sql)

    evaluation_examples.append({
        "predicted_sql": predicted_sql,
        "expected_sql": expected_sql,
    })


# ============================================================
# FINAL METRICS
# ============================================================

if evaluation_examples:
    results = evaluator.evaluate(evaluation_examples)

    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)

    print("Total:", results["total"])
    print("Exact Match Accuracy:", results["exact_match_accuracy"])
    print("Execution Validity:", results["execution_validity"])
    print("Execution Accuracy:", results["execution_accuracy"])


# connection.close()

print("\n===== UNSEEN GENERALIZATION TEST =====")

generalization_tests = {
    "SELECT": [
        ("return every employee record", "SELECT * FROM employees;"),
        ("retrieve the full product list", "SELECT * FROM products;"),
        ("return every order record", "SELECT * FROM orders;"),
    ],

    "EQUAL": [
        ("employees aged exactly 30", "SELECT * FROM employees WHERE age = 30;"),
        ("products costing exactly 50", "SELECT * FROM products WHERE price = 50;"),
        ("orders with exactly 2 items", "SELECT * FROM orders WHERE quantity = 2;"),
    ],

    "GREATER THAN": [
        ("employees over 30 years old", "SELECT * FROM employees WHERE age > 30;"),
        ("products priced higher than 50", "SELECT * FROM products WHERE price > 50;"),
        ("orders containing more than 2 items", "SELECT * FROM orders WHERE quantity > 2;"),
    ],

    "LESS THAN": [
        ("employees below 30 years of age", "SELECT * FROM employees WHERE age < 30;"),
        ("products priced under 50", "SELECT * FROM products WHERE price < 50;"),
        ("orders containing fewer than 2 items", "SELECT * FROM orders WHERE quantity < 2;"),
    ],

    "COUNT": [
        ("tell me the number of employees", "SELECT COUNT(*) FROM employees;"),
        ("how many product records exist", "SELECT COUNT(*) FROM products;"),
        ("what is the total number of orders", "SELECT COUNT(*) FROM orders;"),
    ],

    "AVG": [
        ("mean salary across employees", "SELECT AVG(salary) FROM employees;"),
        ("average price across all products", "SELECT AVG(price) FROM products;"),
        ("mean quantity per order", "SELECT AVG(quantity) FROM orders;"),
    ],

    "DISTINCT": [
        ("which departments appear in employees", "SELECT DISTINCT department FROM employees;"),
        ("what product categories are present", "SELECT DISTINCT category FROM products;"),
        ("which order statuses occur", "SELECT DISTINCT status FROM orders;"),
    ],

    "ORDER BY ASC": [
        ("rank employees from lowest salary to highest", "SELECT * FROM employees ORDER BY salary ASC;"),
        ("arrange products from cheapest to most expensive", "SELECT * FROM products ORDER BY price ASC;"),
        ("arrange orders from smallest quantity to largest", "SELECT * FROM orders ORDER BY quantity ASC;"),
    ],

    "ORDER BY DESC": [
        ("rank employees from highest salary to lowest", "SELECT * FROM employees ORDER BY salary DESC;"),
        ("arrange products from most expensive to cheapest", "SELECT * FROM products ORDER BY price DESC;"),
        ("arrange orders from largest quantity to smallest", "SELECT * FROM orders ORDER BY quantity DESC;"),
    ],

    "LIMIT": [
        ("return only 5 employees", "SELECT * FROM employees LIMIT 5;"),
        ("retrieve 5 product records", "SELECT * FROM products LIMIT 5;"),
        ("give me 5 order records", "SELECT * FROM orders LIMIT 5;"),
    ],
    "IN": [
        (
            "employees who are either 26 or 31 years old",
            "SELECT * FROM employees WHERE age IN (26, 31);"
        ),
        (
            "products costing either 45 or 120",
            "SELECT * FROM products WHERE price IN (45, 120);"
        ),
        (
            "orders containing either 1 or 3 items",
            "SELECT * FROM orders WHERE quantity IN (1, 3);"
        ),
    ],
}

generalization_results = {}

for capability, tests in generalization_tests.items():
    correct = 0

    print(f"\n--- {capability} ---")

    for question, expected_sql in tests:
        try:
            generated_sql = pipeline.generate(question)
            passed = generated_sql.strip() == expected_sql.strip()

            if passed:
                correct += 1

            print("✓" if passed else "✗", question)

            if not passed:
                print("  Expected :", expected_sql)
                print("  Generated:", generated_sql)

        except Exception as error:
            print("✗", question)
            print("  Expected:", expected_sql)
            print("  Error   :", error)

    generalization_results[capability] = (correct, len(tests))


print("\n===== GENERALIZATION SUMMARY =====")

total_correct = 0
total_tests = 0

for capability, (correct, total) in generalization_results.items():
    total_correct += correct
    total_tests += total

    print(f"{capability:15} {correct}/{total}")

print("------------------------------")
print(f"Overall         {total_correct}/{total_tests}")
print(f"Accuracy        {total_correct / total_tests:.2%}")
print("==================================")

connection.close()
