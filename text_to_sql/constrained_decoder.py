from text_to_sql.sql_grammar import SQLGrammar
from text_to_sql.execution_validator import ExecutionValidator


class ConstrainedDecoder:
    """Grammar/schema validity only; natural-language semantics live in the model."""

    def __init__(self,tokenizer,schema,grammar,execution_validator,value_candidates=None,table_candidates=None):
        if tokenizer is None:raise ValueError("tokenizer cannot be None.")
        if not isinstance(schema,dict) or not schema:raise ValueError("schema must be a non-empty dictionary.")
        if not isinstance(grammar,SQLGrammar):raise TypeError("grammar must be an SQLGrammar.")
        if not isinstance(execution_validator,ExecutionValidator):raise TypeError("execution_validator must be an ExecutionValidator.")
        self.tokenizer=tokenizer
        self.schema=schema
        self.grammar=grammar
        self.execution_validator=execution_validator
        self.value_candidates=list(value_candidates or [])
        self.table_candidates=list(table_candidates or [])

        self.selected_table=None
        self.selected_columns=[]
        self.projection_columns=[]
        self.projection_family=None
        self.where_column=None
        self.order_column=None
        self.selected_operator=None
        self.selected_direction=None
        self.in_value_index=0
        # Track numeric literal occurrences already consumed by scalar SQL
        # roles. A literal may be reused only when it appears multiple times
        # in the question. This prevents a WHERE value from being silently
        # recycled as LIMIT while remaining schema/question agnostic.
        self.value_usage={}

        # Historical token-level candidate API compatibility.
        self.active_token_type=None
        self.active_candidates={}
        self.generated_candidate_ids=[]

    def clone(self):
        grammar=SQLGrammar();grammar.state=self.grammar.state
        other=ConstrainedDecoder(
            self.tokenizer,self.schema,grammar,self.execution_validator,
            self.value_candidates,self.table_candidates,
        )
        for name in (
            "selected_table","projection_family","where_column","order_column",
            "selected_operator","selected_direction","in_value_index","active_token_type"
        ):
            setattr(other,name,getattr(self,name))
        other.selected_columns=list(self.selected_columns)
        other.projection_columns=list(self.projection_columns)
        other.value_usage=dict(self.value_usage)
        other.active_candidates={k:list(v) for k,v in self.active_candidates.items()}
        other.generated_candidate_ids=list(self.generated_candidate_ids)
        return other

    def _encode_candidate(self,candidate):
        ids=self.tokenizer.encode_ids(candidate)
        if not ids:return []
        if self.tokenizer.vocab["<UNK>"] in ids:return []
        return ids

    def _get_table_candidates(self):
        tables=[t for t in self.table_candidates if t in self.schema] or list(self.schema)
        if self.projection_columns:
            tables=[t for t in tables if all(c in self.schema[t] for c in self.projection_columns)]
        return tables

    def _get_column_candidates(self):
        if self.selected_table is not None:
            columns=list(self.schema[self.selected_table])
        else:
            columns=[]
            for table in self._get_table_candidates():
                for column in self.schema[table]:
                    if column not in columns:columns.append(column)
        if self.grammar.state in (SQLGrammar.AFTER_SELECT,SQLGrammar.AFTER_DISTINCT,SQLGrammar.AFTER_AVG_OPEN):
            columns=[c for c in columns if c not in self.projection_columns]
        return columns

    def get_allowed_token_types(self):
        allowed=self.grammar.allowed_token_types()
        if self.grammar.state==SQLGrammar.AFTER_SELECT and self.projection_columns:
            return ["COLUMN"]
        if self.grammar.state==SQLGrammar.IN_SELECT_LIST:
            if self.projection_family in {"STAR","COUNT","AVG","DISTINCT"}:
                return ["FROM"]
            if len(self.projection_columns)>=2:
                return ["FROM"]
        if self.grammar.state in (SQLGrammar.AFTER_IN_OPEN,SQLGrammar.AFTER_IN_COMMA):
            if self.in_value_index>=len(self.value_candidates):return []
        if self.grammar.state==SQLGrammar.AFTER_IN_VALUE:
            return [",",")"] if self.in_value_index<len(self.value_candidates) else [")"]
        return allowed

    def _available_value_candidates(self):
        available=[]
        seen={}
        for value in self.value_candidates:
            seen[value]=seen.get(value,0)+1
            if self.value_usage.get(value,0)<seen[value]:
                available.append(value)
        return available

    def get_candidates_for_type(self,token_type):
        if token_type=="TABLE":return self._get_table_candidates()
        if token_type=="COLUMN":return self._get_column_candidates()
        fixed={
            "SELECT":["SELECT "],"*":["*"],",":[", "],"FROM":[" FROM "],";":[";"],
            "WHERE":[" WHERE "],"ORDER BY":[" ORDER BY "],"LIMIT":[" LIMIT "],
            "DISTINCT":["DISTINCT "],"COUNT":["COUNT"],"AVG":["AVG"],"(":["("],")":[")"],
            "OPERATOR":[" = "," > "," < "," IN "],"DIRECTION":[" ASC"," DESC"],
        }
        if token_type in fixed:return fixed[token_type]
        if token_type=="VALUE":
            available=self._available_value_candidates()
            if self.grammar.state in (
                SQLGrammar.AFTER_IN_OPEN,
                SQLGrammar.AFTER_IN_COMMA,
            ):
                return available[:1]
            return available
        raise ValueError(f"Unsupported token type: {token_type}")

    def commit(self,token_type,candidate):
        before=self.grammar.state
        self.grammar.consume(token_type,candidate=candidate)
        if token_type=="VALUE" and before in {
            SQLGrammar.AFTER_OPERATOR,
            SQLGrammar.AFTER_LIMIT,
            SQLGrammar.AFTER_IN_OPEN,
            SQLGrammar.AFTER_IN_COMMA,
        }:
            self.value_usage[candidate]=(
                self.value_usage.get(candidate,0)+1
            )
        if token_type=="COLUMN":
            self.selected_columns.append(candidate)
            if before in (SQLGrammar.AFTER_SELECT,SQLGrammar.AFTER_DISTINCT,SQLGrammar.AFTER_AVG_OPEN):
                self.projection_columns.append(candidate)
                if self.projection_family is None:self.projection_family="COLUMN"
            elif before==SQLGrammar.AFTER_WHERE:self.where_column=candidate
            elif before==SQLGrammar.AFTER_ORDER_BY:self.order_column=candidate
        elif token_type=="*":
            if before==SQLGrammar.AFTER_SELECT:self.projection_family="STAR"
        elif token_type=="DISTINCT":self.projection_family="DISTINCT"
        elif token_type=="COUNT":self.projection_family="COUNT"
        elif token_type=="AVG":self.projection_family="AVG"
        elif token_type=="TABLE":self.selected_table=candidate
        elif token_type=="OPERATOR":self.selected_operator=candidate.strip().upper()
        elif token_type=="DIRECTION":self.selected_direction=candidate.strip().upper()
        elif token_type=="VALUE" and self.grammar.state==SQLGrammar.AFTER_IN_VALUE:
            self.in_value_index+=1

    def validate_selected_schema(self):
        if self.selected_table is None or not self.execution_validator.table_exists(self.selected_table):return False
        return all(self.execution_validator.column_exists_in_table(self.selected_table,c) for c in self.selected_columns)

    # Historical candidate API.
    def get_candidate_token_sequences(self,token_type):
        return {c:self._encode_candidate(c) for c in self.get_candidates_for_type(token_type) if self._encode_candidate(c)}

    def start_candidate(self,token_type):
        if self.active_token_type is not None:raise RuntimeError("A candidate is already active.")
        if token_type not in self.get_allowed_token_types():raise ValueError("Token type is not allowed.")
        self.active_token_type=token_type
        self.active_candidates=self.get_candidate_token_sequences(token_type)
        self.generated_candidate_ids=[]

    def get_allowed_next_token_ids(self):
        if not self.active_candidates:return []
        n=len(self.generated_candidate_ids)
        out=[]
        for ids in self.active_candidates.values():
            if n<len(ids) and ids[n] not in out:out.append(ids[n])
        return out

    def consume_candidate_token(self,token_id):
        if token_id not in self.get_allowed_next_token_ids():raise ValueError(f"Token ID {token_id} is not allowed.")
        self.generated_candidate_ids.append(token_id)
        prefix=self.generated_candidate_ids
        self.active_candidates={c:ids for c,ids in self.active_candidates.items() if ids[:len(prefix)]==prefix}

    def is_candidate_complete(self):
        return any(ids==self.generated_candidate_ids for ids in self.active_candidates.values())

    def get_completed_candidate(self):
        for candidate,ids in self.active_candidates.items():
            if ids==self.generated_candidate_ids:return candidate
        return None

    def reset_candidate_tracking(self):
        self.active_token_type=None;self.active_candidates={};self.generated_candidate_ids=[]

    def complete_candidate(self):
        candidate=self.get_completed_candidate()
        if candidate is None:raise RuntimeError("No completed candidate.")
        token_type=self.active_token_type
        self.reset_candidate_tracking()
        self.commit(token_type,candidate)
        return candidate

    @property
    def finished(self):
        return self.grammar.state==SQLGrammar.COMPLETE
