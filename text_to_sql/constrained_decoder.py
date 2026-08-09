from text_to_sql.sql_grammar import SQLGrammar
from text_to_sql.execution_validator import ExecutionValidator

class ConstrainedDecoder:
    def __init__(self,tokenizer,schema: dict[str, list[str]],grammar: SQLGrammar,execution_validator: ExecutionValidator
                 , value_candidates: list[str] | None = None) -> None:
        if tokenizer is None:
            raise ValueError("tokenizer cannot be None.")
        if not isinstance(schema, dict):
            raise TypeError("schema must be a dictionary.")
        if not schema:
            raise ValueError("schema cannot be empty.")
        if not isinstance(grammar, SQLGrammar):
            raise TypeError("grammar must be an SQLGrammar.")
        if not isinstance(execution_validator,ExecutionValidator):
            raise TypeError("execution_validator must be an ExecutionValidator.")
        self.tokenizer = tokenizer
        self.schema = schema
        self.grammar = grammar
        self.execution_validator = execution_validator
        self.active_token_type: str | None = None
        self.active_candidates: dict[str, list[int]] = {}
        self.generated_candidate_ids: list[int] = []
        self.selected_columns: list[str] = []
        self.selected_table: str | None = None
        self.value_candidates = value_candidates or []

    def _get_table_candidates(self) -> list[str]:
        return list(self.schema.keys())

    def _get_column_candidates(self) -> list[str]:
        columns = []
        for table_columns in self.schema.values():
            for column in table_columns:
                if column not in columns:
                    columns.append(column)
        return columns

    def get_candidates_for_type(self,token_type: str) -> list[str]:
        if token_type == "TABLE":
            return self._get_table_candidates()
        if token_type == "COLUMN":
            return self._get_column_candidates()

        fixed_candidates = {
            "SELECT": ["SELECT "],
            "*": ["*"],
            ",": [", "],
            "FROM": [" FROM "],
            ";": [";"],
            "WHERE": [" WHERE "],
            "ORDER BY": [" ORDER BY "],
            "LIMIT": [" LIMIT "],
            "DISTINCT": ["DISTINCT "],
            "COUNT": ["COUNT"],
            "AVG": ["AVG"],
            "(": ["("],
            ")": [")"],
            "OPERATOR": [" = ", " > ", " < "],
            "DIRECTION": [" ASC", " DESC"],
        }

        if token_type in fixed_candidates:
            return fixed_candidates[token_type]

        if token_type == "VALUE":
            return self.value_candidates

        raise ValueError(f"Unsupported token type: {token_type}")

    def _encode_candidate(self,candidate: str) -> list[int]:
        token_ids = self.tokenizer.encode_ids(candidate)
        if not token_ids:
            raise ValueError(f"Candidate could not be encoded: {candidate}")
        return token_ids

    def get_candidate_token_sequences(self,token_type: str) -> dict[str, list[int]]:
        candidates = self.get_candidates_for_type(token_type)
        sequences = {}
        for candidate in candidates:
            sequences[candidate] = self._encode_candidate(candidate)
        return sequences

    def get_allowed_next_token_ids(self) -> list[int]:
        if not self.active_candidates:
            return []

        prefix_length = len(self.generated_candidate_ids)
        allowed_token_ids = []
        for token_ids in self.active_candidates.values():
            if prefix_length >= len(token_ids):
                continue
            next_token_id = token_ids[prefix_length]
            if next_token_id not in allowed_token_ids:
                allowed_token_ids.append(next_token_id)

        return allowed_token_ids

    def consume_candidate_token(self,token_id: int) -> None:

        if not isinstance(token_id, int):
            raise TypeError("token_id must be an integer.")

        allowed_token_ids = (self.get_allowed_next_token_ids())
        if token_id not in allowed_token_ids:
            raise ValueError(f"Token ID {token_id} is not allowed.")
        self.generated_candidate_ids.append(token_id)
        prefix = self.generated_candidate_ids
        remaining_candidates = {}
        for candidate, token_ids in (self.active_candidates.items()):
            if token_ids[:len(prefix)] == prefix:
                remaining_candidates[candidate] = token_ids
        self.active_candidates = remaining_candidates

    def is_candidate_complete(self) -> bool:
        if not self.active_candidates:
            return False
        for token_ids in self.active_candidates.values():
            if token_ids == self.generated_candidate_ids:
                return True
        return False

    def get_completed_candidate(self) -> str | None:
        for candidate, token_ids in (self.active_candidates.items()):
            if token_ids == self.generated_candidate_ids:
                return candidate
        return None

    def reset_candidate_tracking(self) -> None:
        self.active_token_type = None
        self.active_candidates = {}
        self.generated_candidate_ids = []

    def start_candidate(self,token_type: str) -> None:
        if self.active_token_type is not None:
            raise RuntimeError("A candidate is already active.")
        allowed_types = self.grammar.allowed_token_types()
        if token_type not in allowed_types:
            raise ValueError(f"{token_type} is not allowed by the current grammar state.")
        self.active_token_type = token_type
        self.active_candidates = self.get_candidate_token_sequences(token_type)

        self.generated_candidate_ids = []

    def complete_candidate(self) -> str:
        completed_candidate = self.get_completed_candidate()
        if completed_candidate is None:
            raise RuntimeError("No candidate has been completed.")
        token_type = self.active_token_type
        if token_type is None:
            raise RuntimeError("No active token type.")
        self.grammar.consume(token_type)
        self._record_completed_candidate(token_type=token_type,candidate=completed_candidate)
        self.reset_candidate_tracking()
        return completed_candidate

    def mask_logits(self,logits: list[float],allowed_token_ids: list[int]) -> list[float]:
        if not logits:
            raise ValueError("logits cannot be empty.")

        if not allowed_token_ids:
            raise ValueError("allowed_token_ids cannot be empty.")
        masked_logits = [float("-inf") for _ in logits]
        for token_id in allowed_token_ids:
            if token_id < 0 or token_id >= len(logits):
                raise ValueError(f"Token ID {token_id} is outside logits range.")
            masked_logits[token_id] = logits[token_id]
        return masked_logits

    def select_best_allowed_token(self,logits: list[float]) -> int:

        allowed_token_ids = self.get_allowed_next_token_ids()
        masked_logits = self.mask_logits(logits=logits,allowed_token_ids=allowed_token_ids)
        best_token_id = max(range(len(masked_logits)),key=lambda token_id: masked_logits[token_id])

        return best_token_id

    def generate_next_token(self,model,input_ids: list[int]) -> int:
        if not input_ids:
            raise ValueError("input_ids cannot be empty.")
        logits = model.forward(input_ids)
        last_logits = logits[-1]
        token_id = self.select_best_allowed_token(last_logits)
        self.consume_candidate_token(token_id)
        return token_id

    def generate_candidate(self,model,input_ids: list[int],token_type: str) -> tuple[str, list[int]]:
        self.start_candidate(token_type)
        generated_ids = []
        while not self.is_candidate_complete():
            token_id = self.generate_next_token(model=model,input_ids=input_ids)
            generated_ids.append(token_id)
            input_ids.append(token_id)
        completed_candidate = (self.complete_candidate())
        return completed_candidate, generated_ids

    def _record_completed_candidate(self,token_type: str,candidate: str) -> None:

        if token_type == "COLUMN":
            self.selected_columns.append(candidate)

        elif token_type == "TABLE":
            self.selected_table = candidate

    def validate_selected_schema(self) -> bool:

        if self.selected_table is None:
            return False
        if not self.execution_validator.table_exists(self.selected_table):
            return False
        for column in self.selected_columns:
            if not (self.execution_validator.column_exists_in_table(self.selected_table,column)):
                return False

        return True

    def rank_allowed_tokens(self,logits: list[float]) -> list[int]:

        allowed_token_ids = self.get_allowed_next_token_ids()
        if not allowed_token_ids:
            return []
        return sorted(allowed_token_ids,key=lambda token_id: logits[token_id],reverse=True)

    def score_candidate(self,model,input_ids: list[int],candidate_ids: list[int]) -> float:

        working_input_ids = input_ids.copy()
        total_score = 0.0

        for token_id in candidate_ids:
            logits = model.forward(working_input_ids)
            last_logits = logits[-1]
            total_score += last_logits[token_id]
            working_input_ids.append(token_id)

        return total_score

    def rank_candidates(self,model,input_ids: list[int],token_type: str) -> list[tuple[str, float]]:
        candidate_sequences = (self.get_candidate_token_sequences(token_type))
        scored_candidates = []
        for candidate, candidate_ids in (candidate_sequences.items()):
            score = self.score_candidate(model=model,input_ids=input_ids,candidate_ids=candidate_ids)
            scored_candidates.append((candidate, score))
        scored_candidates.sort(key=lambda item: item[1],reverse=True)
        return scored_candidates

    def select_valid_table(self,model,input_ids: list[int]) -> str:
        ranked_tables = self.rank_candidates(model=model,input_ids=input_ids,token_type="TABLE")
        for table, score in ranked_tables:
            valid = True
            for column in self.selected_columns:
                if not self.execution_validator.column_exists_in_table(table,column):
                    valid = False
                    break
            if valid:
                return table
        raise RuntimeError("No valid table found for the selected columns.")

    def commit_table(self,table: str,input_ids: list[int],) -> list[int]:
        if not self.execution_validator.table_exists(table):
            raise ValueError(f"Unknown table: {table}")
        table_ids = self._encode_candidate(table)
        input_ids.extend(table_ids)
        self.selected_table = table
        self.grammar.consume("TABLE")
        return table_ids

    def clone(self):
        grammar_copy = SQLGrammar()
        grammar_copy.state = self.grammar.state
        decoder_copy = ConstrainedDecoder(tokenizer=self.tokenizer,schema=self.schema,grammar=grammar_copy,
                                          execution_validator=self.execution_validator, value_candidates=self.value_candidates.copy())
        decoder_copy.active_token_type = self.active_token_type
        decoder_copy.active_candidates = {candidate: token_ids.copy() for candidate, token_ids in self.active_candidates.items()}
        decoder_copy.generated_candidate_ids = (self.generated_candidate_ids.copy())
        decoder_copy.selected_columns = (self.selected_columns.copy())
        decoder_copy.selected_table = self.selected_table
        return decoder_copy