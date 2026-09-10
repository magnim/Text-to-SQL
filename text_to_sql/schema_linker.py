import re


class SchemaLinker:
    def __init__(self, schema: dict[str, list[str]]):
        self.schema = schema

    def link_tables(self, question: str) -> list[str]:
        normalized_question = self._normalize(question)
        matched_tables = []

        for table_name in self.schema:
            normalized_table = self._normalize(table_name)
            pattern = rf"\b{re.escape(normalized_table)}\b"

            if re.search(pattern, normalized_question):
                matched_tables.append(table_name)

        return matched_tables

    def _normalize(self, text: str) -> str:
        return text.lower().replace("_", " ").strip()


if __name__ == "__main__":
    schema = {
        "employees": ["id", "name", "department", "salary", "age", "years_experience"],
        "products": ["id", "name", "category", "price", "stock", "rating"],
        "orders": ["id", "product_id", "employee_id", "quantity", "total_amount", "status"],
        "customers": ["id", "name", "city", "age", "membership_level"],
    }

    linker = SchemaLinker(schema)

    test_cases = [
        ("show all customers", ["customers"]),
        ("show customers", ["customers"]),
        ("SHOW ALL CUSTOMERS", ["customers"]),
        ("show all employees", ["employees"]),
        ("show products", ["products"]),
        ("show all orders", ["orders"]),
        ("show records older than 30", []),

        # Boundary / normalization checks
        ("show preorders", []),
        ("show clients", []),
        ("show order items", ["order_items"]),
    ]

    for question, expected in test_cases:
        result = linker.link_tables(question)

        print("Question:", question)
        print("Expected:", expected)
        print("Actual  :", result)
        print()

        assert result == expected

    print("SchemaLinker tests passed.")