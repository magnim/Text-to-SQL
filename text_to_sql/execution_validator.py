import sqlite3
try:
    import pymysql
    MYSQL_ERRORS = (pymysql.MySQLError,)
except ImportError:
    MYSQL_ERRORS = ()
from text_to_sql.sql_hypothesis import SQLHypothesis


class ExecutionValidator:

    def __init__(self,schema: dict[str, list[str]],connection=None) -> None:
        if not isinstance(schema, dict):
            raise TypeError("schema must be a dictionary.")
        if not schema:
            raise ValueError("schema cannot be empty.")

        self.schema = schema
        self.connection = connection

    def table_exists(self,table: str) -> bool:
        return table in self.schema

    def column_exists_in_table(self,table: str,column: str) -> bool:
        if not self.table_exists(table):
            return False
        return column in self.schema[table]

    def validate_sql(self,sql: str) -> bool:
        if not isinstance(sql, str):
            raise TypeError("sql must be a string.")
        if not sql.strip():
            raise ValueError("sql cannot be empty.")
        if self.connection is None:
            raise RuntimeError("Database connection is not available.")
        try:
            cursor = self.connection.cursor()
            cursor.execute(sql)
            return True
        except (sqlite3.Error,) + MYSQL_ERRORS:
            return False

    def validate_hypothesis(self,hypothesis: SQLHypothesis) -> bool:
        result = self.validate_sql(hypothesis.sql)
        hypothesis.execution_valid = result
        return result

    def select_best_hypothesis(self,hypotheses: list[SQLHypothesis]) -> SQLHypothesis:
        if not hypotheses:
            raise ValueError("hypotheses cannot be empty.")
        ranked_hypotheses = sorted(hypotheses,key=lambda hypothesis: hypothesis.score,reverse=True)
        for hypothesis in ranked_hypotheses:
            if self.validate_hypothesis(hypothesis):return hypothesis
        raise RuntimeError("No executable SQL hypothesis found.")