import sqlite3

from from_scratch.models.tiny_gpt import TinyGPT
from training.dataset import build_text_to_sql_training_dataset
from from_scratch.training.train_tiny_gpt import train_tiny_gpt

from text_to_sql.schema_encoder import SchemaEncoder
from text_to_sql.prompt_builder import PromptBuilder
from training.text_to_sql_dataset import build_text_to_sql_corpus
from text_to_sql.text_to_sql_pipeline import TextToSQLPipeline
from text_to_sql.sql_evaluator import SQLEvaluator


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

cursor.execute("""
CREATE TABLE employees (
    id INTEGER,
    name TEXT,
    department TEXT,
    salary REAL,
    age INTEGER,
    years_experience INTEGER
)
""")

cursor.execute("""
CREATE TABLE products (
    id INTEGER,
    name TEXT,
    category TEXT,
    price REAL,
    stock INTEGER,
    rating REAL
)
""")

cursor.execute("""
CREATE TABLE orders (
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
    prompt_builder=prompt_builder,
)

print("Training examples:", len(corpus))


# ============================================================
# TOKENIZER + TRAINING DATASET
# ============================================================

tokenizer, training_dataset = build_text_to_sql_training_dataset(
    corpus=corpus,schema=schema,
    num_merges=100,
    # window_size=13,
    # stride=6,
)

print("Vocabulary size:", len(tokenizer.vocab))
print("Training samples:", len(training_dataset))


# ============================================================
# MODEL
# ============================================================
max_length = max(len(example["input_ids"]) for example in training_dataset)
print(f"Longest training sequence  : {max_length}")
maximum_sequence_length = max_length + 32

model = TinyGPT(
    vocabulary_size=len(tokenizer.vocab),
    embedding_dimension=8,
    maximum_sequence_length=maximum_sequence_length,
    number_of_heads=2,
    hidden_dimension=16,
    number_of_layers=2,
)


# ============================================================
# TRAIN MODEL
# ============================================================

epoch_losses = train_tiny_gpt(
    model=model,
    training_dataset=training_dataset,
    epochs=20,
    learning_rate=0.005,
    batch_size=2,
)


# ============================================================
# TEXT-to-SQL PIPELINE
# ============================================================

pipeline = TextToSQLPipeline(
    model=model,
    tokenizer=tokenizer,
    schema=schema,
    connection=connection,
    beam_width=3,
)


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

test_cases = [
    {
        "question": "show all employees",
        "expected_sql": "SELECT * FROM employees;",
    },
    {
        "question": "how many products are there",
        "expected_sql": "SELECT COUNT(*) FROM products;",
    },
    {
        "question": "show all orders",
        "expected_sql": "SELECT * FROM orders;",
    },
    {
        "question": "show employees with salary greater than 60000",
        "expected_sql": "SELECT * FROM employees WHERE salary > 60000;",
    },
    {
        "question": "what is the average salary",
        "expected_sql": "SELECT AVG(salary) FROM employees;",
    },
    {
        "question": "show distinct departments",
        "expected_sql": "SELECT DISTINCT department FROM employees;",
    },
    {
        "question": "show the top 5 products",
        "expected_sql": "SELECT * FROM products LIMIT 5;",
    },
    {
        "question": "show employees ordered by salary from highest",
        "expected_sql": "SELECT * FROM employees ORDER BY salary DESC;",
    },
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


connection.close()