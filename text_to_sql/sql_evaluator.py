import sqlite3


class SQLEvaluator:

    def __init__(self, connection) -> None:
        if connection is None:
            raise ValueError("connection cannot be None.")

        self.connection = connection

    def exact_match(self, predicted_sql: str, expected_sql: str) -> bool:
        predicted = predicted_sql.strip().lower()
        expected = expected_sql.strip().lower()

        return predicted == expected

    def execution_valid(self, sql: str) -> bool:
        try:
            cursor = self.connection.cursor()
            cursor.execute(sql)
            return True
        except sqlite3.Error:
            return False

    def execution_match(self, predicted_sql: str, expected_sql: str) -> bool:
        try:
            cursor = self.connection.cursor()

            cursor.execute(predicted_sql)
            predicted_result = cursor.fetchall()

            cursor.execute(expected_sql)
            expected_result = cursor.fetchall()

            return predicted_result == expected_result

        except sqlite3.Error:
            return False

    def evaluate(self, examples: list[dict]) -> dict:
        total = len(examples)

        if total == 0:
            raise ValueError("examples cannot be empty.")

        exact_matches = 0
        valid_queries = 0
        execution_matches = 0

        for example in examples:
            predicted_sql = example["predicted_sql"]
            expected_sql = example["expected_sql"]

            if self.exact_match(predicted_sql, expected_sql):
                exact_matches += 1

            if self.execution_valid(predicted_sql):
                valid_queries += 1

            if self.execution_match(predicted_sql, expected_sql):
                execution_matches += 1

        return {
            "total": total,
            "exact_match_accuracy": exact_matches / total,
            "execution_validity": valid_queries / total,
            "execution_accuracy": execution_matches / total,
        }