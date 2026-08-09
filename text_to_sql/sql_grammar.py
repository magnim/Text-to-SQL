class SQLGrammar:

    START = 0
    AFTER_SELECT = 1
    IN_SELECT_LIST = 2
    AFTER_FROM = 3
    AFTER_TABLE = 4
    COMPLETE = 5
    AFTER_WHERE = 6
    AFTER_WHERE_COLUMN = 7
    AFTER_OPERATOR = 8
    AFTER_VALUE = 9
    AFTER_ORDER_BY = 10
    AFTER_ORDER_COLUMN = 11
    AFTER_DIRECTION = 12
    AFTER_LIMIT = 13
    AFTER_LIMIT_VALUE = 14
    AFTER_DISTINCT = 15
    AFTER_COUNT = 16
    AFTER_COUNT_OPEN = 17
    AFTER_COUNT_STAR = 18
    AFTER_AVG = 19
    AFTER_AVG_OPEN = 20
    AFTER_AVG_COLUMN = 21

    def __init__(self) -> None:
        self.state = self.START

    def allowed_token_types(self) -> list[str]:

        if self.state == self.START:
            return ["SELECT"]
        if self.state == self.AFTER_SELECT:
            return ["COLUMN", "*", "DISTINCT","COUNT","AVG"]
        if self.state == self.IN_SELECT_LIST:
            return [",", "FROM"]
        if self.state == self.AFTER_DISTINCT:
            return ["COLUMN"]
        if self.state == self.AFTER_FROM:
            return ["TABLE"]
        if self.state == self.AFTER_TABLE:
            return [";","WHERE","ORDER BY","LIMIT"]
        if self.state == self.AFTER_WHERE:
            return ["COLUMN"]

        if self.state == self.AFTER_WHERE_COLUMN:
            return ["OPERATOR"]

        if self.state == self.AFTER_OPERATOR:
            return ["VALUE"]
        if self.state == self.AFTER_ORDER_BY:
            return ["COLUMN"]

        if self.state == self.AFTER_ORDER_COLUMN:
            return ["DIRECTION"]

        if self.state == self.AFTER_DIRECTION:
            return [";"]

        if self.state == self.AFTER_VALUE:
            return [";"]
        if self.state == self.COMPLETE:
            return []
        if self.state == self.AFTER_LIMIT:
            return ["VALUE"]

        if self.state == self.AFTER_LIMIT_VALUE:
            return [";"]
        if self.state == self.AFTER_COUNT:
            return ["("]

        if self.state == self.AFTER_COUNT_OPEN:
            return ["*"]

        if self.state == self.AFTER_COUNT_STAR:
            return [")"]
        if self.state == self.AFTER_AVG:
            return ["("]

        if self.state == self.AFTER_AVG_OPEN:
            return ["COLUMN"]

        if self.state == self.AFTER_AVG_COLUMN:
            return [")"]
        raise RuntimeError(f"Unknown SQL grammar state: {self.state}")

    def consume(self,token_type: str) -> None:
        allowed_types = self.allowed_token_types()
        if token_type not in allowed_types:
            raise ValueError(f"{token_type} is not allowed in state {self.state}.")
        if self.state == self.START:
            self.state = self.AFTER_SELECT
            return
        if self.state == self.AFTER_SELECT:
            if token_type == "DISTINCT":
                self.state = self.AFTER_DISTINCT
            elif token_type == "COUNT":
                self.state = self.AFTER_COUNT
            elif token_type == "AVG":
                self.state = self.AFTER_AVG

            else:
                self.state = self.IN_SELECT_LIST
            return
        if self.state == self.IN_SELECT_LIST:
            if token_type == ",":
                self.state = self.AFTER_SELECT
            elif token_type == "FROM":
                self.state = self.AFTER_FROM
            return
        if self.state == self.AFTER_FROM:
            self.state = self.AFTER_TABLE
            return
        if self.state == self.AFTER_TABLE:
            if token_type == ";":
                self.state = self.COMPLETE
            elif token_type == "WHERE":
                self.state = self.AFTER_WHERE
            elif token_type == "ORDER BY":
                self.state = self.AFTER_ORDER_BY
            elif token_type == "LIMIT":
                self.state = self.AFTER_LIMIT
            return
        if self.state == self.AFTER_WHERE:
            self.state = self.AFTER_WHERE_COLUMN
            return

        if self.state == self.AFTER_WHERE_COLUMN:
            self.state = self.AFTER_OPERATOR
            return

        if self.state == self.AFTER_OPERATOR:
            self.state = self.AFTER_VALUE
            return

        if self.state == self.AFTER_VALUE:
            self.state = self.COMPLETE
            return
        if self.state == self.AFTER_ORDER_BY:
            self.state = self.AFTER_ORDER_COLUMN
            return

        if self.state == self.AFTER_ORDER_COLUMN:
            self.state = self.AFTER_DIRECTION
            return

        if self.state == self.AFTER_DIRECTION:
            self.state = self.COMPLETE
            return
        if self.state == self.AFTER_LIMIT:
            self.state = self.AFTER_LIMIT_VALUE
            return

        if self.state == self.AFTER_LIMIT_VALUE:
            self.state = self.COMPLETE
            return
        if self.state == self.AFTER_DISTINCT:
            self.state = self.IN_SELECT_LIST
            return
        if self.state == self.AFTER_COUNT:
            self.state = self.AFTER_COUNT_OPEN
            return

        if self.state == self.AFTER_COUNT_OPEN:
            self.state = self.AFTER_COUNT_STAR
            return

        if self.state == self.AFTER_COUNT_STAR:
            self.state = self.IN_SELECT_LIST
            return
        if self.state == self.AFTER_AVG:
            self.state = self.AFTER_AVG_OPEN
            return

        if self.state == self.AFTER_AVG_OPEN:
            self.state = self.AFTER_AVG_COLUMN
            return

        if self.state == self.AFTER_AVG_COLUMN:
            self.state = self.IN_SELECT_LIST
            return