class SQLHypothesis:

    def __init__(self,token_ids: list[int],sql: str,score: float) -> None:

        self.token_ids = token_ids
        self.sql = sql
        self.score = score
        self.execution_valid: bool | None = None