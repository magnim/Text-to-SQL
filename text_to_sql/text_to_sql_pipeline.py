import re
import torch

from text_to_sql.schema_role_encoder import SchemaRoleEncoder
from text_to_sql.sql_grammar import SQLGrammar
from text_to_sql.constrained_decoder import ConstrainedDecoder
from text_to_sql.execution_validator import ExecutionValidator
from text_to_sql.sql_beam import SQLBeam,SQLBeamSearch


class AmbiguousSchemaError(RuntimeError):
    pass


class TextToSQLPipeline:
    """Schema-conditioned Transformer Text-to-SQL inference."""

    def __init__(
        self,model,tokenizer,schema,connection,beam_width=12,
        grounding_weight=1.0,structure_weight=1.0,
        semantic_schema_encoder=None,schema_semantic_weight=4.0,
        schema_semantic_order_weight=0.25,generation_token_reserve=96,
        column_types=None,**legacy
    ):
        if model is None:raise ValueError("model cannot be None.")
        if tokenizer is None:raise ValueError("tokenizer cannot be None.")
        if not isinstance(schema,dict) or not schema:raise ValueError("schema must be a non-empty dictionary.")
        self.model=model
        self.tokenizer=tokenizer
        self.schema=schema
        self.connection=connection
        self.beam_width=beam_width
        self.grounding_weight=grounding_weight
        self.structure_weight=structure_weight
        self.semantic_schema_encoder=semantic_schema_encoder
        self.schema_semantic_weight=float(schema_semantic_weight)
        self.schema_semantic_order_weight=float(schema_semantic_order_weight)
        self.generation_token_reserve=int(generation_token_reserve)
        self.column_types=dict(column_types or {})

        # Legacy compatibility attributes; intentionally non-authoritative.
        self.semantic_column_linker=None
        self.schema_linker=None
        self.semantic_weight=0.0
        self.operator_weight=0.0
        self.ambiguity_detection=False

    @staticmethod
    def _normalize_identifier(text):
        text=re.sub(r"(?<=[a-z0-9])(?=[A-Z])"," ",text)
        text=text.replace("_"," ").lower()
        return " ".join(text.split())

    def _mentioned_prompt_table(self,question,semantic_scores):
        normalized_question=self._normalize_identifier(question)
        mentioned=[]
        for table in self.schema:
            normalized_table=self._normalize_identifier(table)
            if re.search(rf"(?<!\w){re.escape(normalized_table)}(?!\w)",normalized_question):
                mentioned.append(table)
        if not mentioned:
            return None
        table_scores=semantic_scores.get("tables",{})
        return max(mentioned,key=lambda table:float(table_scores.get(table,0.0)))

    def _select_prompt_table(self,question,semantic_scores):
        mentioned=self._mentioned_prompt_table(question,semantic_scores)
        if mentioned is not None:
            return mentioned
        table_scores=semantic_scores.get("tables",{})
        if table_scores:
            return max(table_scores,key=table_scores.get)
        return None

    @staticmethod
    def _comparison_type_family(type_name):
        normalized=str(type_name or "").upper()
        if any(token in normalized for token in ("INT","REAL","FLOA","DOUB","DEC","NUM")):
            return "numeric"
        if any(token in normalized for token in ("DATE","TIME","YEAR")):
            return "temporal"
        if any(token in normalized for token in ("CHAR","CLOB","TEXT","BLOB","BINARY","JSON","ENUM","SET")):
            return "text"
        return "unknown"

    def _apply_numeric_comparison_type_prior(self,semantic_scores,intents,values,question):
        if not values or not self.column_types:
            return
        operator_probs=torch.exp(torch.tensor(intents["operator"],dtype=torch.float32))
        if float(max(operator_probs[1],operator_probs[2]))<0.70:
            return

        where_scores=dict(
            semantic_scores.get("where_columns",semantic_scores.get("columns",{}))
        )
        if not where_scores:
            return

        normalized_question=self._normalize_identifier(question)
        directly_mentioned=set()
        for key in where_scores:
            column=self._normalize_identifier(key[1])
            if re.search(rf"(?<!\w){re.escape(column)}(?!\w)",normalized_question):
                directly_mentioned.add(key)

        year_like_value=False
        if len(values)==1:
            try:
                numeric_value=float(values[0])
                year_like_value=numeric_value.is_integer() and 1000<=numeric_value<=2999
            except ValueError:
                pass

        for key,score in list(where_scores.items()):
            family=self._comparison_type_family(self.column_types.get(key))
            if family=="text":
                where_scores[key]=float(score)-1.0
                continue
            if family in {"numeric","temporal"}:
                adjusted=2.0*float(score)
                if year_like_value and not directly_mentioned:
                    column_tokens=set(self._normalize_identifier(key[1]).split())
                    if family=="temporal" or "year" in column_tokens:
                        adjusted+=0.30
                where_scores[key]=adjusted
        semantic_scores["where_columns"]=where_scores

    def _predict_intents(self,input_ids,role_ids,spans):
        q_start,q_end=spans["question"]
        question_token_ids=input_ids[q_start:q_end]
        with torch.no_grad():
            hidden=self.model.get_hidden_states(input_ids,role_ids)[0]
            out=self.model.predict_intents_from_hidden(hidden,spans["question"])
            semantic=self.model.predict_semantic_structure_from_hidden(
                hidden,spans["question"]
            )
            lexical=self.model.predict_lexical_structure(
                question_token_ids
            )
            # Preserve the verified pretrained operator classifier at the
            # active schema-conditioned prompt boundary.
            operator=self.model.predict_operator(input_ids,role_ids)[0]

        legacy_clause=torch.sigmoid(out["clause_logits"])
        semantic_clause=torch.sigmoid(semantic["clause_logits"])
        lexical_clause=torch.sigmoid(lexical["clause_logits"])
        clause_probs=(
            0.10*legacy_clause
            +0.10*semantic_clause
            +0.80*lexical_clause
        )

        legacy_cont=torch.softmax(out["continuation_logits"],dim=-1)
        semantic_cont=torch.softmax(semantic["continuation_logits"],dim=-1)
        lexical_cont=torch.softmax(
            lexical["continuation_logits"],dim=-1
        )
        continuation_probs=(
            0.10*legacy_cont
            +0.10*semantic_cont
            +0.80*lexical_cont
        )


        return {
            "projection":torch.log_softmax(out["projection_logits"],dim=-1).tolist(),
            "arity":torch.log_softmax(out["projection_arity_logits"],dim=-1).tolist(),
            "clauses":clause_probs.tolist(),
            "lexical_clauses":lexical_clause.tolist(),
            "operator":torch.log_softmax(operator,dim=-1).tolist(),
            "direction":torch.log_softmax(out["direction_logits"],dim=-1).tolist(),
            "continuation":torch.log(
                continuation_probs.clamp_min(1e-8)
            ).tolist(),
            "lexical_continuation":lexical_cont.tolist(),
        }

    def generate(self,question):
        if not isinstance(question,str):raise TypeError("question must be a string.")
        if not question.strip():raise ValueError("question cannot be empty.")

        encoder=SchemaRoleEncoder(self.tokenizer)
        semantic_scores={"tables":{},"columns":{}}
        if self.semantic_schema_encoder is not None:
            semantic_scores=self.semantic_schema_encoder.score_schema(
                self.tokenizer,question,self.schema
            )

        prompt_schema=self.schema
        prompt_table=None
        input_ids,role_ids,spans=encoder.encode_prompt_with_spans(prompt_schema,question)
        prompt_budget=self.model.maximum_sequence_length-self.generation_token_reserve
        if len(input_ids)>prompt_budget:
            prompt_table=self._select_prompt_table(question,semantic_scores)
            if prompt_table is None:
                raise ValueError("Schema/question prompt leaves insufficient SQL generation headroom.")
            prompt_schema={prompt_table:self.schema[prompt_table]}
            input_ids,role_ids,spans=encoder.encode_prompt_with_spans(prompt_schema,question)
            if len(input_ids)>prompt_budget:
                raise ValueError("Selected table schema leaves insufficient SQL generation headroom.")

        intents=self._predict_intents(input_ids,role_ids,spans)
        values=re.findall(r"\b\d+(?:\.\d+)?\b",question)

        if self.semantic_schema_encoder is not None:
            # The generic semantic encoder is more stable for a single WHERE
            # target because it is not competing with an absent ORDER role.
            # Keep the role-specific WHERE scores only when the learned lexical
            # structure head says ORDER BY is part of the question, where role
            # separation is required to distinguish WHERE from ORDER columns.
            lexical_clauses=intents.get("lexical_clauses")
            if (
                lexical_clauses is not None
                and float(lexical_clauses[1]) < 0.50
                and semantic_scores.get("columns")
            ):
                semantic_scores["where_columns"]=semantic_scores["columns"]

            # Operator intent describes the comparison in the question, not
            # unrelated database tables. The pretrained operator head was
            # trained with schema context, so keep that context but restrict
            # it to the semantically selected table to prevent full-schema
            # contamination from flipping >/< /= decisions.
            table_scores=semantic_scores.get("tables",{})
            if table_scores:
                operator_table=prompt_table or max(table_scores,key=table_scores.get)
                operator_schema={operator_table:self.schema[operator_table]}
                operator_ids,operator_roles,_=encoder.encode_prompt_with_spans(
                    operator_schema,question
                )
                if len(operator_ids)<self.model.maximum_sequence_length:
                    with torch.no_grad():
                        operator=self.model.predict_operator(
                            operator_ids,operator_roles
                        )[0]
                    intents["operator"]=torch.log_softmax(
                        operator,dim=-1
                    ).tolist()

            # Numeric inequality values should ground to columns whose declared
            # database types can sensibly participate in numeric/temporal
            # comparisons. This is a schema-type prior, not a phrase mapping.
            self._apply_numeric_comparison_type_prior(
                semantic_scores,intents,values,question
            )

            if values and semantic_scores["columns"]:
                best_column=max(semantic_scores["columns"].values())
                operator_probs=torch.exp(torch.tensor(intents["operator"],dtype=torch.float32))
                comparison_prob=float(operator_probs[1:].max())
                if best_column>=0.72 and comparison_prob>=0.70:
                    intents["clauses"][0]=max(float(intents["clauses"][0]),0.985)

        # A single numeric literal cannot simultaneously satisfy independent
        # WHERE and LIMIT roles. Use the schema-invariant learned structure
        # head to resolve that competition before beam search. This is model
        # confidence fusion only; there are no phrase/column mappings here.
        if len(values)==1:
            lexical_clauses=intents.get("lexical_clauses")
            lexical_cont=intents.get("lexical_continuation")
            if lexical_clauses is not None and lexical_cont is not None:
                where_p=float(lexical_clauses[0])
                limit_p=float(lexical_clauses[2])

                if where_p>=0.90 and where_p>limit_p+0.02:
                    intents["clauses"][0]=max(
                        float(intents["clauses"][0]),0.995
                    )
                    intents["clauses"][2]=min(
                        float(intents["clauses"][2]),0.005
                    )
                    continuation=torch.exp(torch.tensor(
                        intents["continuation"],dtype=torch.float32
                    ))
                    continuation[2]=min(
                        float(continuation[2]),1e-6
                    )
                    continuation[3]=min(
                        float(continuation[3]),1e-6
                    )
                    continuation=continuation/continuation.sum().clamp_min(1e-8)
                    intents["continuation"]=torch.log(
                        continuation.clamp_min(1e-8)
                    ).tolist()

                elif limit_p>=0.90 and limit_p>where_p+0.10:
                    intents["clauses"][2]=max(
                        float(intents["clauses"][2]),0.995
                    )
                    intents["clauses"][0]=min(
                        float(intents["clauses"][0]),0.005
                    )
                    continuation=torch.exp(torch.tensor(
                        intents["continuation"],dtype=torch.float32
                    ))
                    continuation[0]=min(
                        float(continuation[0]),1e-6
                    )
                    continuation[1]=min(
                        float(continuation[1]),1e-6
                    )
                    continuation[3]=min(
                        float(continuation[3]),1e-6
                    )
                    continuation[2]=max(
                        float(continuation[2]),0.995
                    )
                    continuation=continuation/continuation.sum().clamp_min(1e-8)
                    intents["continuation"]=torch.log(
                        continuation.clamp_min(1e-8)
                    ).tolist()

        validator=ExecutionValidator(self.schema,self.connection)
        decoder=ConstrainedDecoder(
            self.tokenizer,self.schema,SQLGrammar(),validator,
            value_candidates=values,
            table_candidates=[prompt_table] if prompt_table is not None else [],
        )
        initial=SQLBeam(input_ids,role_ids,0.0,decoder,len(input_ids),0)

        qs,qe=spans["question"]
        search=SQLBeamSearch(
            self.model,decoder,beam_width=self.beam_width,
            schema_spans=spans,question_token_ids=input_ids[qs:qe],
            projection_log_probabilities=intents["projection"],
            projection_arity_log_probabilities=intents["arity"],
            clause_probabilities=intents["clauses"],
            operator_log_probabilities=intents["operator"],
            direction_log_probabilities=intents["direction"],
            continuation_log_probabilities=intents["continuation"],
            grounding_weight=self.grounding_weight,
            structure_weight=self.structure_weight,
            schema_semantic_scores=semantic_scores,
            schema_semantic_weight=self.schema_semantic_weight,
            schema_semantic_order_weight=self.schema_semantic_order_weight,
        )
        best=search.select_best_executable(search.search(initial))
        return self.tokenizer.decode_ids(best.token_ids[len(input_ids):]).strip()
