class SQLGrammar:
    START=0
    AFTER_SELECT=1
    IN_SELECT_LIST=2
    AFTER_FROM=3
    AFTER_TABLE=4
    COMPLETE=5
    AFTER_WHERE=6
    AFTER_WHERE_COLUMN=7
    AFTER_OPERATOR=8
    AFTER_VALUE=9
    AFTER_ORDER_BY=10
    AFTER_ORDER_COLUMN=11
    AFTER_DIRECTION=12
    AFTER_LIMIT=13
    AFTER_LIMIT_VALUE=14
    AFTER_DISTINCT=15
    AFTER_COUNT=16
    AFTER_COUNT_OPEN=17
    AFTER_COUNT_STAR=18
    AFTER_AVG=19
    AFTER_AVG_OPEN=20
    AFTER_AVG_COLUMN=21
    AFTER_IN_OPERATOR=22
    AFTER_IN_OPEN=23
    AFTER_IN_VALUE=24
    AFTER_IN_COMMA=25

    def __init__(self):
        self.state=self.START

    def allowed_token_types(self):
        s=self.state
        if s==self.START:return ["SELECT"]
        if s==self.AFTER_SELECT:return ["COLUMN","*","DISTINCT","COUNT","AVG"]
        if s==self.IN_SELECT_LIST:return [",","FROM"]
        if s==self.AFTER_DISTINCT:return ["COLUMN"]
        if s==self.AFTER_COUNT:return ["("]
        if s==self.AFTER_COUNT_OPEN:return ["*"]
        if s==self.AFTER_COUNT_STAR:return [")"]
        if s==self.AFTER_AVG:return ["("]
        if s==self.AFTER_AVG_OPEN:return ["COLUMN"]
        if s==self.AFTER_AVG_COLUMN:return [")"]
        if s==self.AFTER_FROM:return ["TABLE"]
        if s==self.AFTER_TABLE:return [";","WHERE","ORDER BY","LIMIT"]
        if s==self.AFTER_WHERE:return ["COLUMN"]
        if s==self.AFTER_WHERE_COLUMN:return ["OPERATOR"]
        if s==self.AFTER_OPERATOR:return ["VALUE"]
        if s==self.AFTER_IN_OPERATOR:return ["("]
        if s==self.AFTER_IN_OPEN:return ["VALUE"]
        if s==self.AFTER_IN_VALUE:return [",",")"]
        if s==self.AFTER_IN_COMMA:return ["VALUE"]
        if s==self.AFTER_VALUE:return [";","ORDER BY","LIMIT"]
        if s==self.AFTER_ORDER_BY:return ["COLUMN"]
        if s==self.AFTER_ORDER_COLUMN:return ["DIRECTION"]
        if s==self.AFTER_DIRECTION:return [";","LIMIT"]
        if s==self.AFTER_LIMIT:return ["VALUE"]
        if s==self.AFTER_LIMIT_VALUE:return [";"]
        if s==self.COMPLETE:return []
        raise RuntimeError(f"Unknown SQL grammar state: {s}")

    def consume(self,token_type,candidate=None):
        if token_type not in self.allowed_token_types():
            raise ValueError(f"{token_type} is not allowed in state {self.state}.")
        s=self.state
        if s==self.START:self.state=self.AFTER_SELECT
        elif s==self.AFTER_SELECT:
            if token_type=="DISTINCT":self.state=self.AFTER_DISTINCT
            elif token_type=="COUNT":self.state=self.AFTER_COUNT
            elif token_type=="AVG":self.state=self.AFTER_AVG
            else:self.state=self.IN_SELECT_LIST
        elif s==self.AFTER_DISTINCT:self.state=self.IN_SELECT_LIST
        elif s==self.IN_SELECT_LIST:self.state=self.AFTER_SELECT if token_type=="," else self.AFTER_FROM
        elif s==self.AFTER_COUNT:self.state=self.AFTER_COUNT_OPEN
        elif s==self.AFTER_COUNT_OPEN:self.state=self.AFTER_COUNT_STAR
        elif s==self.AFTER_COUNT_STAR:self.state=self.IN_SELECT_LIST
        elif s==self.AFTER_AVG:self.state=self.AFTER_AVG_OPEN
        elif s==self.AFTER_AVG_OPEN:self.state=self.AFTER_AVG_COLUMN
        elif s==self.AFTER_AVG_COLUMN:self.state=self.IN_SELECT_LIST
        elif s==self.AFTER_FROM:self.state=self.AFTER_TABLE
        elif s==self.AFTER_TABLE:
            self.state={
                ";":self.COMPLETE,
                "WHERE":self.AFTER_WHERE,
                "ORDER BY":self.AFTER_ORDER_BY,
                "LIMIT":self.AFTER_LIMIT,
            }[token_type]
        elif s==self.AFTER_WHERE:self.state=self.AFTER_WHERE_COLUMN
        elif s==self.AFTER_WHERE_COLUMN:
            self.state=self.AFTER_IN_OPERATOR if candidate and candidate.strip().upper()=="IN" else self.AFTER_OPERATOR
        elif s==self.AFTER_OPERATOR:self.state=self.AFTER_VALUE
        elif s==self.AFTER_IN_OPERATOR:self.state=self.AFTER_IN_OPEN
        elif s==self.AFTER_IN_OPEN:self.state=self.AFTER_IN_VALUE
        elif s==self.AFTER_IN_VALUE:self.state=self.AFTER_IN_COMMA if token_type=="," else self.AFTER_VALUE
        elif s==self.AFTER_IN_COMMA:self.state=self.AFTER_IN_VALUE
        elif s==self.AFTER_VALUE:
            self.state={
                ";":self.COMPLETE,
                "ORDER BY":self.AFTER_ORDER_BY,
                "LIMIT":self.AFTER_LIMIT,
            }[token_type]
        elif s==self.AFTER_ORDER_BY:self.state=self.AFTER_ORDER_COLUMN
        elif s==self.AFTER_ORDER_COLUMN:self.state=self.AFTER_DIRECTION
        elif s==self.AFTER_DIRECTION:self.state=self.COMPLETE if token_type==";" else self.AFTER_LIMIT
        elif s==self.AFTER_LIMIT:self.state=self.AFTER_LIMIT_VALUE
        elif s==self.AFTER_LIMIT_VALUE:self.state=self.COMPLETE
