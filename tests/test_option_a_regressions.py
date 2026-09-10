
import sqlite3
from pathlib import Path

import pytest
import torch

from pytorch_impl.tiny_gpt import TinyGPT
from src.tokenizer.bpe import BPETrainer
from text_to_sql.text_to_sql_pipeline import TextToSQLPipeline


SCHEMA = {
    "employees": ["id", "name", "department", "salary", "age", "years_experience"],
    "products": ["id", "name", "category", "price", "stock", "rating"],
    "orders": ["id", "product_id", "employee_id", "quantity", "total_amount", "status"],
}


@pytest.fixture(scope="module")
def pipeline():
    root = Path(__file__).resolve().parents[1]
    connection = sqlite3.connect(":memory:")
    cursor = connection.cursor()
    cursor.execute(
        "CREATE TABLE employees "
        "(id INTEGER, name TEXT, department TEXT, salary REAL, age INTEGER, years_experience INTEGER)"
    )
    cursor.execute(
        "CREATE TABLE products "
        "(id INTEGER, name TEXT, category TEXT, price REAL, stock INTEGER, rating REAL)"
    )
    cursor.execute(
        "CREATE TABLE orders "
        "(id INTEGER, product_id INTEGER, employee_id INTEGER, quantity INTEGER, total_amount REAL, status TEXT)"
    )
    connection.commit()

    tokenizer = BPETrainer.load(str(root / "tokenizer_balanced.json"))

    model = TinyGPT(
        vocabulary_size=len(tokenizer.vocab),
        embedding_dimension=64,
        maximum_sequence_length=512,
        number_of_heads=4,
        hidden_dimension=128,
        number_of_layers=4,
        number_of_schema_roles=5,
    )
    model.load_state_dict(
        torch.load(
            root / "tiny_gpt_schema_aware_final.pt",
            map_location="cpu",
            weights_only=True,
        ),
        strict=True,
    )
    model.eval()

    result = TextToSQLPipeline(
        model=model,
        tokenizer=tokenizer,
        schema=SCHEMA,
        connection=connection,
        beam_width=12,
    )
    yield result
    connection.close()


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        (
            "show employees where salary is greater than 60000",
            "SELECT * FROM employees WHERE salary > 60000;",
        ),
        (
            "retrieve the full product list",
            "SELECT * FROM products;",
        ),
        (
            "which order statuses occur",
            "SELECT DISTINCT status FROM orders;",
        ),
        (
            "rank employees from lowest salary to highest",
            "SELECT * FROM employees ORDER BY salary ASC;",
        ),
        (
            "arrange orders from smallest quantity to largest",
            "SELECT * FROM orders ORDER BY quantity ASC;",
        ),
        (
            "arrange products from most expensive to cheapest",
            "SELECT * FROM products ORDER BY price DESC;",
        ),
        (
            "arrange orders from largest quantity to smallest",
            "SELECT * FROM orders ORDER BY quantity DESC;",
        ),
        # This currently passes on the uploaded source and must not regress
        # while operator-aware beam ranking is corrected.
        (
            "employees aged exactly 30",
            "SELECT * FROM employees WHERE age = 30;",
        ),
        (
            "find employees under 30 years old",
            "SELECT * FROM employees WHERE age < 30;",
        ),
        (
            "orders containing more than 2 items",
            "SELECT * FROM orders WHERE quantity > 2;",
        ),
        (
            "return only 5 employees",
            "SELECT * FROM employees LIMIT 5;",
        ),
        (
            "retrieve 5 product records",
            "SELECT * FROM products LIMIT 5;",
        ),
        (
            "give me 5 order records",
            "SELECT * FROM orders LIMIT 5;",
        ),
        (
            "show employee names and salaries",
            "SELECT name, salary FROM employees;",
        ),
        (
            "show product names and prices",
            "SELECT name, price FROM products;",
        ),
        (
            "show order quantities and statuses",
            "SELECT quantity, status FROM orders;",
        ),
        (
            "show employee names and departments",
            "SELECT name, department FROM employees;",
        ),
        (
            "show employee names and salaries where age is greater than 30",
            "SELECT name, salary FROM employees WHERE age > 30;",
        ),
        (
            "show employee names and salaries where age equals 30",
            "SELECT name, salary FROM employees WHERE age = 30;",
        ),
        (
            "show product names and prices where price is less than 100",
            "SELECT name, price FROM products WHERE price < 100;",
        ),
        (
            "show order quantities and statuses where quantity is greater than 2",
            "SELECT quantity, status FROM orders WHERE quantity > 2;",
        ),
    ],
)
def test_option_a_real_model_regressions(pipeline, question, expected):
    assert pipeline.generate(question).strip() == expected


def test_unseen_current_schema_where_linking(pipeline):
    connection = sqlite3.connect(":memory:")
    connection.execute(
        "CREATE TABLE customers (id INTEGER, name TEXT, age INTEGER)"
    )
    connection.commit()

    unseen_pipeline = TextToSQLPipeline(
        model=pipeline.model,
        tokenizer=pipeline.tokenizer,
        schema={"customers": ["id", "name", "age"]},
        connection=connection,
        beam_width=12,
    )

    try:
        assert unseen_pipeline.generate(
            "show customers where age is greater than 30"
        ).strip() == "SELECT * FROM customers WHERE age > 30;"
    finally:
        connection.close()



def test_unseen_current_schema_older_than_maps_to_greater_than(pipeline):
    connection = sqlite3.connect(":memory:")
    connection.execute(
        "CREATE TABLE customers (id INTEGER, name TEXT, city TEXT, age INTEGER, membership_level TEXT)"
    )
    connection.commit()

    unseen_pipeline = TextToSQLPipeline(
        model=pipeline.model,
        tokenizer=pipeline.tokenizer,
        schema={"customers": ["id", "name", "city", "age", "membership_level"]},
        connection=connection,
        beam_width=12,
    )

    try:
        assert unseen_pipeline.generate(
            "show customers older than 30"
        ).strip() == "SELECT * FROM customers WHERE age > 30;"
    finally:
        connection.close()


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        (
            "show all students older than 15",
            "SELECT * FROM students WHERE Age > 15;",
        ),
        (
            "show students older than 15",
            "SELECT * FROM students WHERE Age > 15;",
        ),
    ],
)
def test_unseen_students_older_than_keeps_row_projection_and_filter(
    pipeline,
    question,
    expected,
):
    connection = sqlite3.connect(":memory:")
    connection.execute(
        "CREATE TABLE students (ID INTEGER, Name TEXT, Age INTEGER)"
    )
    connection.commit()

    unseen_pipeline = TextToSQLPipeline(
        model=pipeline.model,
        tokenizer=pipeline.tokenizer,
        schema={"students": ["ID", "Name", "Age"]},
        connection=connection,
        beam_width=12,
    )

    try:
        assert unseen_pipeline.generate(question).strip() == expected
    finally:
        connection.close()
