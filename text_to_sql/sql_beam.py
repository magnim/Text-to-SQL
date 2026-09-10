import math
from dataclasses import dataclass
import torch

from text_to_sql.schema_role_encoder import SchemaRoleEncoder
from text_to_sql.sql_grammar import SQLGrammar


@dataclass
class SQLBeam:
    token_ids:list[int]
    schema_role_ids:list[int]
    score:float
    decoder:object
    prompt_length:int
    scored_token_count:int=0
    grounding_score:float=0.0
    grounding_decisions:int=0
    finished:bool=False
    execution_valid:bool|None=None

    def normalized_score(self):
        return self.score/max(self.scored_token_count,1)


class SQLBeamSearch:
    PROJECTION_INDEX={"STAR":0,"COLUMN":1,"DISTINCT":2,"COUNT":3,"AVG":4}
    OPERATOR_INDEX={None:0,">":1,"<":2,"=":3,"IN":4}
    DIRECTION_INDEX={None:0,"ASC":1,"DESC":2}

    def __init__(
        self,model,decoder,beam_width=12,schema_spans=None,question_token_ids=None,
        projection_log_probabilities=None,projection_arity_log_probabilities=None,
        clause_probabilities=None,operator_log_probabilities=None,direction_log_probabilities=None,
        continuation_log_probabilities=None,grounding_weight=1.0,structure_weight=1.0,
        operator_weight=1.0,contextual_projection_weight=1.0,lexical_column_weight=0.5,
        table_semantic_weight=1.5,where_column_weight=2.5,order_column_weight=2.5,
        normalized_identifier_weight=4.0,embedding_affinity_weight=0.25,
        schema_semantic_scores=None,schema_semantic_weight=4.0,
        schema_semantic_order_weight=0.25,**legacy
    ):
        self.model=model
        self.decoder=decoder
        self.beam_width=int(beam_width)
        self.schema_spans=schema_spans or {"tables":{},"columns":{}}
        self.question_token_ids=list(question_token_ids or [])
        self.projection_log_probabilities=projection_log_probabilities
        self.projection_arity_log_probabilities=projection_arity_log_probabilities
        self.clause_probabilities=clause_probabilities
        self.operator_log_probabilities=operator_log_probabilities
        self.direction_log_probabilities=direction_log_probabilities
        self.continuation_log_probabilities=continuation_log_probabilities
        self.grounding_weight=grounding_weight
        self.structure_weight=structure_weight
        self.operator_weight=operator_weight
        self.contextual_projection_weight=contextual_projection_weight
        self.lexical_column_weight=lexical_column_weight
        self.table_semantic_weight=table_semantic_weight
        self.where_column_weight=where_column_weight
        self.order_column_weight=order_column_weight
        self.normalized_identifier_weight=normalized_identifier_weight
        self.embedding_affinity_weight=embedding_affinity_weight
        self.schema_semantic_scores=schema_semantic_scores or {"tables":{},"columns":{}}
        self.schema_semantic_weight=float(schema_semantic_weight)
        self.schema_semantic_order_weight=float(schema_semantic_order_weight)
        self._cache={}

    @staticmethod
    def _relative(values):
        if values is None:return None
        best=max(values)
        return [float(v)-float(best) for v in values]

    @staticmethod
    def _safe_log(p):
        p=max(min(float(p),1-1e-8),1e-8)
        return math.log(p)

    @staticmethod
    def _log_softmax_list(values):
        if not values:return []
        return torch.log_softmax(torch.tensor(values,dtype=torch.float32),dim=-1).tolist()

    def _hidden_and_log_probs(self,ids,roles):
        key=(tuple(ids),tuple(roles))
        if key in self._cache:return self._cache[key]
        with torch.no_grad():
            hidden=self.model.get_hidden_states(ids,roles)
            logits=self.model.output_projection(hidden)
            log_probs=torch.log_softmax(logits[0,-1],dim=-1).detach()
        result=(hidden[0],log_probs)
        if len(self._cache)>2048:self._cache.clear()
        self._cache[key]=result
        return result

    def _score_candidate_tokens(self,beam,candidate_ids):
        ids=list(beam.token_ids);roles=list(beam.schema_role_ids);score=0.0
        for tid in candidate_ids:
            _,lp=self._hidden_and_log_probs(ids,roles)
            score+=float(lp[tid])
            ids.append(tid);roles.append(SchemaRoleEncoder.SQL)
        return score,ids,roles

    def _projection_family(self,decoder):
        if decoder is None:return "OTHER"
        return getattr(decoder,"projection_family",None) or "OTHER"

    def _active_projection_family(self,beam):
        family=self._projection_family(beam.decoder)
        if family!="OTHER":return family
        decoder=beam.decoder
        if decoder is None:return "OTHER"
        tokenizer=getattr(decoder,"tokenizer",None)
        if tokenizer is not None:
            try:
                prefix=tokenizer.decode_ids(beam.token_ids[beam.prompt_length:]).strip().upper()
                if prefix.startswith("SELECT AVG("):return "AVG"
                if prefix.startswith("SELECT COUNT("):return "COUNT"
                if prefix.startswith("SELECT DISTINCT "):return "DISTINCT"
                if prefix.startswith("SELECT *"):return "STAR"
                if prefix.startswith("SELECT "):return "COLUMN"
            except Exception:pass
        return {
            "*":"STAR","COLUMN":"COLUMN","DISTINCT":"DISTINCT","COUNT":"COUNT","AVG":"AVG"
        }.get(getattr(decoder,"active_token_type",None),"OTHER")

    def _arity(self,decoder):
        family=self._projection_family(decoder)
        if family in {"STAR","COUNT"}:return 0
        if decoder is None:return 0
        columns=list(getattr(decoder,"projection_columns",[]))
        if len(columns)>=2:return 2
        state=getattr(getattr(decoder,"grammar",None),"state",None)
        if len(columns)==1 and state==SQLGrammar.AFTER_SELECT:return 2
        if len(columns)==1 or family in {"COLUMN","DISTINCT","AVG"}:return 1
        return 0

    def _clause_presence(self,decoder):
        if decoder is None:return False,False,False
        state=getattr(getattr(decoder,"grammar",None),"state",None)
        where=getattr(decoder,"where_column",None) is not None or getattr(decoder,"selected_operator",None) is not None
        order=getattr(decoder,"order_column",None) is not None or getattr(decoder,"selected_direction",None) is not None
        # LIMIT has no explicit field, infer by grammar path.
        limit=state in {SQLGrammar.AFTER_LIMIT,SQLGrammar.AFTER_LIMIT_VALUE}
        if state==SQLGrammar.COMPLETE:
            # Decode-independent structural tracking: value candidates used by LIMIT leave no marker.
            # Decoder records a flag when LIMIT is committed if available.
            limit=bool(getattr(decoder,"has_limit",False))
        return where,order,limit

    def _continuation_score(self,beam):
        if self.continuation_log_probabilities is None:return 0.0
        decoder=beam.decoder
        if decoder is None:return 0.0
        where,order,limit=self._clause_presence(decoder)
        if not (where or order or limit):return 0.0
        rel=self._relative(self.continuation_log_probabilities)
        state=getattr(getattr(decoder,"grammar",None),"state",None)
        if order and limit:return rel[3]
        if limit and not order:return rel[2]
        if order and not limit:
            if state==SQLGrammar.COMPLETE:return rel[1]
            # ORDER path may still continue to LIMIT.
            a,b=self.continuation_log_probabilities[1],self.continuation_log_probabilities[3]
            return float(torch.logsumexp(torch.tensor([a,b]),0))-max(self.continuation_log_probabilities)
        if where and state==SQLGrammar.COMPLETE:return rel[0]
        return 0.0

    def _structure_score(self,beam):
        d=beam.decoder
        if d is None:return 0.0
        score=0.0
        family=self._projection_family(d)
        if self.projection_log_probabilities is not None and family in self.PROJECTION_INDEX:
            score+=self._relative(self.projection_log_probabilities)[self.PROJECTION_INDEX[family]]
        if self.projection_arity_log_probabilities is not None:
            score+=2.5*self._relative(
                self.projection_arity_log_probabilities
            )[self._arity(d)]
        if self.clause_probabilities is not None:
            where,order,limit=self._clause_presence(d)
            # WHERE and ORDER auxiliary heads guide branches; LIMIT stays LM-led unless
            # the continuation head specifically supports a composition.
            weights=(2.5,1.0,0.0)
            for i,present in enumerate((where,order,limit)):
                if present and weights[i]:
                    p=self.clause_probabilities[i]
                    score+=weights[i]*(self._safe_log(p)-self._safe_log(1-p))
        op=getattr(d,"selected_operator",None)
        if op is not None and self.operator_log_probabilities is not None:
            score+=self.operator_weight*self._relative(self.operator_log_probabilities)[self.OPERATOR_INDEX.get(op,0)]
        direction=getattr(d,"selected_direction",None)
        if direction is not None and self.direction_log_probabilities is not None:
            score+=0.6*self._relative(self.direction_log_probabilities)[self.DIRECTION_INDEX.get(direction,0)]
        score+=1.5*self._continuation_score(beam)
        return self.structure_weight*score

    def selection_score(self,beam):
        score=beam.normalized_score()+self.grounding_weight*beam.grounding_score
        score+=self._structure_score(beam)
        return score

    def _span_groups(self,beam,token_type,candidates):
        groups=[]
        if token_type=="TABLE":
            for table in candidates:
                spans=[]
                if table in self.schema_spans.get("tables",{}):spans.append(self.schema_spans["tables"][table])
                for column in beam.decoder.schema[table]:
                    span=self.schema_spans.get("columns",{}).get((table,column))
                    if span is not None:spans.append(span)
                groups.append(spans)
        else:
            for column in candidates:
                spans=[]
                if beam.decoder.selected_table is not None:
                    span=self.schema_spans.get("columns",{}).get((beam.decoder.selected_table,column))
                    if span is not None:spans.append(span)
                else:
                    for table,cols in beam.decoder.schema.items():
                        if column in cols:
                            span=self.schema_spans.get("columns",{}).get((table,column))
                            if span is not None:spans.append(span)
                groups.append(spans)
        return groups

    def _lexical_field_groups(self,beam,token_type,candidates):
        if token_type=="TABLE":
            return [
                [beam.decoder._encode_candidate(table)]+[
                    beam.decoder._encode_candidate(c) for c in beam.decoder.schema[table]
                ] for table in candidates
            ]
        return [[beam.decoder._encode_candidate(c)] for c in candidates]

    def _value_grounding_relative(self, beam, candidates, kind):
        if (
            self.model is None
            or not candidates
            or "question" not in self.schema_spans
            or not hasattr(self.model, "score_question_value_spans")
        ):
            return [0.0] * len(candidates)

        q_start, q_end = self.schema_spans["question"]
        question_ids = beam.token_ids[q_start:q_end]
        candidate_spans = []
        for candidate in candidates:
            token_ids = beam.decoder._encode_candidate(candidate)
            spans = []
            if token_ids:
                for offset in range(
                    0,
                    len(question_ids) - len(token_ids) + 1,
                ):
                    if (
                        question_ids[
                            offset:offset + len(token_ids)
                        ] == token_ids
                    ):
                        spans.append(
                            (
                                q_start + offset,
                                q_start + offset + len(token_ids),
                            )
                        )
            candidate_spans.append(spans)

        if not any(candidate_spans):
            return [0.0] * len(candidates)

        hidden, _ = self._hidden_and_log_probs(
            beam.token_ids, beam.schema_role_ids
        )
        with torch.no_grad():
            scores = self.model.score_question_value_spans(
                hidden,
                candidate_spans,
                kind,
            )
        return self._relative(
            torch.log_softmax(scores, dim=-1).tolist()
        )

    def _identifier_grounding_relative(self,beam,token_type,candidates):
        if token_type not in {"TABLE","COLUMN"} or self.model is None:
            return [0.0]*len(candidates)
        hidden,_=self._hidden_and_log_probs(beam.token_ids,beam.schema_role_ids)
        qspan=self.schema_spans.get("question")
        raw=[0.0]*len(candidates)

        if token_type=="TABLE":
            if qspan and hasattr(self.model,"score_schema_fields_from_question"):
                with torch.no_grad():
                    scores=self.model.score_schema_fields_from_question(
                        hidden,qspan,self._lexical_field_groups(beam,token_type,candidates),"table"
                    )
                lp=self._relative(torch.log_softmax(scores,dim=-1).tolist())
                raw=[self.table_semantic_weight*x for x in lp]
            if hasattr(self.model,"score_schema_spans_from_hidden"):
                spans=self._span_groups(beam,token_type,candidates)
                if all(spans):
                    with torch.no_grad():
                        ctx=self.model.score_schema_spans_from_hidden(hidden,-1,spans,"table")
                    rel=self._relative(torch.log_softmax(ctx,dim=-1).tolist())
                    raw=[a+0.25*b for a,b in zip(raw,rel)]

            # Confidence-adaptive Transformer embedding grounding. A table is
            # represented by its own identifier plus all current-schema
            # columns. This is schema-generic and uses no phrase/table map.
            if self.embedding_affinity_weight>0 and hasattr(self.model,"score_identifier_embedding_affinity"):
                vals=[]
                for table in candidates:
                    field_values=[]
                    for field in [table]+list(beam.decoder.schema[table]):
                        with torch.no_grad():
                            field_values.append(float(
                                self.model.score_identifier_embedding_affinity(
                                    self.question_token_ids,
                                    beam.decoder._encode_candidate(field),
                                )
                            ))
                    vals.append(max(field_values) if field_values else 0.0)
                if vals:
                    ordered=sorted(vals,reverse=True)
                    best=ordered[0]
                    second=ordered[1] if len(ordered)>1 else 0.0
                    margin=best-second
                    if best>=0.20:
                        rel=self._relative(self._log_softmax_list([v/0.10 for v in vals]))
                        weight=3.5 if margin>=0.12 else self.embedding_affinity_weight
                        raw=[a+weight*b for a,b in zip(raw,rel)]
            v2_tables=self.schema_semantic_scores.get("tables",{})
            if v2_tables:
                vals=[float(v2_tables.get(t,-1.0)) for t in candidates]
                rel=self._relative(self._log_softmax_list([v/0.08 for v in vals]))
                raw=[a+self.schema_semantic_weight*b for a,b in zip(raw,rel)]
            return self._relative(raw)

        state=getattr(beam.decoder.grammar,"state",None)
        token_lists=[beam.decoder._encode_candidate(c) for c in candidates]

        # Clause-specific semantic column grounding.
        if qspan and state==SQLGrammar.AFTER_WHERE and hasattr(self.model,"score_clause_columns_from_question"):
            with torch.no_grad():
                scores=self.model.score_clause_columns_from_question(
                    hidden,qspan,token_lists,"where"
                )
            rel=self._relative(
                torch.log_softmax(scores,dim=-1).tolist()
            )
            raw=[self.where_column_weight*x for x in rel]

            # WHERE receives additional evidence from the Transformer at the
            # actual generated "WHERE " prefix. This keeps the role-specific
            # semantic head, but lets sequence-conditioned decoder evidence
            # disambiguate compositions containing a different ORDER column.
            spans=self._span_groups(
                beam,token_type,candidates
            )
            if all(spans) and hasattr(
                self.model,
                "score_schema_spans_from_hidden",
            ):
                with torch.no_grad():
                    contextual=self.model.score_schema_spans_from_hidden(
                        hidden,-1,spans,"column"
                    )
                contextual_rel=self._relative(
                    torch.log_softmax(
                        contextual,dim=-1
                    ).tolist()
                )
                raw=[
                    a+1.0*b
                    for a,b in zip(raw,contextual_rel)
                ]
        elif qspan and state==SQLGrammar.AFTER_ORDER_BY and hasattr(self.model,"score_clause_columns_from_question"):
            with torch.no_grad():
                scores=self.model.score_clause_columns_from_question(hidden,qspan,token_lists,"order")
            rel=self._relative(torch.log_softmax(scores,dim=-1).tolist())
            raw=[self.order_column_weight*x for x in rel]
        else:
            # Projection columns: current-prefix pointer + question semantic matching.
            spans=self._span_groups(beam,token_type,candidates)
            if all(spans) and hasattr(self.model,"score_schema_spans_from_hidden"):
                with torch.no_grad():
                    ctx=self.model.score_schema_spans_from_hidden(hidden,-1,spans,"column")
                rel=self._relative(torch.log_softmax(ctx,dim=-1).tolist())
                raw=[self.contextual_projection_weight*x for x in rel]
            if qspan and hasattr(self.model,"score_schema_fields_from_question"):
                with torch.no_grad():
                    sem=self.model.score_schema_fields_from_question(
                        hidden,qspan,self._lexical_field_groups(beam,token_type,candidates),"column"
                    )
                rel=self._relative(torch.log_softmax(sem,dim=-1).tolist())
                raw=[a+self.lexical_column_weight*b for a,b in zip(raw,rel)]

        # Case-normalized model signal for mixed-case schema identifiers.
        if qspan and any(c!=c.lower() for c in candidates) and hasattr(self.model,"score_normalized_columns_from_question"):
            lower=[beam.decoder._encode_candidate(c.lower()) for c in candidates]
            if all(lower):
                with torch.no_grad():
                    norm=self.model.score_normalized_columns_from_question(hidden,qspan,lower)
                rel=self._relative(torch.log_softmax(norm,dim=-1).tolist())
                raw=[a+self.normalized_identifier_weight*b for a,b in zip(raw,rel)]

        # Confidence-adaptive Transformer embedding grounding for columns.
        # When one current-schema identifier has a clear learned-token match
        # to the question, amplify that model signal; otherwise keep it as a
        # weak tie-breaker. No natural-language phrase rules are used.
        if self.embedding_affinity_weight>0 and hasattr(self.model,"score_identifier_embedding_affinity"):
            vals=[]
            for c in candidates:
                with torch.no_grad():
                    vals.append(float(self.model.score_identifier_embedding_affinity(
                        self.question_token_ids,beam.decoder._encode_candidate(c)
                    )))
            if vals:
                ordered=sorted(vals,reverse=True)
                best=ordered[0]
                second=ordered[1] if len(ordered)>1 else 0.0
                margin=best-second
                if best>=0.20:
                    rel=self._relative(self._log_softmax_list([v/0.10 for v in vals]))
                    semantic_clause_state = state in {
                        SQLGrammar.AFTER_WHERE,
                        SQLGrammar.AFTER_ORDER_BY,
                    }
                    later_projection_state = (
                        state == SQLGrammar.AFTER_SELECT
                        and bool(getattr(beam.decoder, "projection_columns", []))
                    )
                    weight=(
                        4.0
                        if (
                            (semantic_clause_state or later_projection_state)
                            and margin>=0.12
                        )
                        else self.embedding_affinity_weight
                    )
                    raw=[a+weight*b for a,b in zip(raw,rel)]

        if state==SQLGrammar.AFTER_WHERE:
            v2_columns=self.schema_semantic_scores.get("where_columns",self.schema_semantic_scores.get("columns",{}))
        elif state==SQLGrammar.AFTER_ORDER_BY:
            v2_columns=self.schema_semantic_scores.get("order_columns",self.schema_semantic_scores.get("columns",{}))
        else:
            v2_columns=self.schema_semantic_scores.get("columns",{})
        if v2_columns and state in {SQLGrammar.AFTER_WHERE,SQLGrammar.AFTER_ORDER_BY}:
            vals=[]
            for c in candidates:
                if beam.decoder.selected_table is not None:
                    vals.append(float(v2_columns.get((beam.decoder.selected_table,c),-1.0)))
                else:
                    matches=[float(v2_columns.get((t,c),-1.0)) for t,cols in beam.decoder.schema.items() if c in cols]
                    vals.append(max(matches) if matches else -1.0)
            rel=self._relative(self._log_softmax_list([v/0.08 for v in vals]))
            # V2 semantic grounding remains strong for WHERE/table selection.
            # ORDER BY uses a smaller contribution so clause-conditioned
            # TinyGPT evidence can separate the ORDER target from a different
            # WHERE target in composed questions without phrase-specific rules.
            semantic_weight=(
                self.schema_semantic_order_weight
                if state==SQLGrammar.AFTER_ORDER_BY
                else self.schema_semantic_weight
            )
            raw=[a+semantic_weight*b for a,b in zip(raw,rel)]

        return self._relative(raw)

    def _expand_beam(self,beam):
        if beam.finished:return [beam]
        children=[]
        for token_type in beam.decoder.get_allowed_token_types():
            candidates=beam.decoder.get_candidates_for_type(token_type)
            if not candidates:continue
            if token_type in {"TABLE", "COLUMN"}:
                grounding=self._identifier_grounding_relative(
                    beam,token_type,candidates
                )
                grounding_kind=True
            elif (
                token_type == "VALUE"
                and beam.decoder.grammar.state
                in {
                    SQLGrammar.AFTER_OPERATOR,
                    SQLGrammar.AFTER_LIMIT,
                }
            ):
                value_kind=(
                    "where"
                    if beam.decoder.grammar.state
                    == SQLGrammar.AFTER_OPERATOR
                    else "limit"
                )
                grounding=self._value_grounding_relative(
                    beam,candidates,value_kind
                )
                grounding_kind=True
            else:
                grounding=[0.0]*len(candidates)
                grounding_kind=False

            for candidate,gscore in zip(candidates,grounding):
                ids=beam.decoder._encode_candidate(candidate)
                if not ids:continue
                lm,new_ids,new_roles=self._score_candidate_tokens(beam,ids)
                decoder=beam.decoder.clone()
                try:decoder.commit(token_type,candidate)
                except ValueError:continue
                # Structural flags used after grammar completion.
                if token_type=="LIMIT":decoder.has_limit=True
                elif not hasattr(decoder,"has_limit"):decoder.has_limit=getattr(beam.decoder,"has_limit",False)
                child=SQLBeam(
                    new_ids,new_roles,beam.score+lm,decoder,beam.prompt_length,
                    beam.scored_token_count+len(ids),
                    beam.grounding_score+(gscore if grounding_kind else 0.0),
                    beam.grounding_decisions+int(grounding_kind),
                )
                child.finished=decoder.grammar.state==SQLGrammar.COMPLETE
                children.append(child)
        return children

    def expand_beam(self,beam):
        d=beam.decoder
        if d is None or hasattr(d,"get_allowed_token_types"):
            return self._expand_beam(beam)
        # Historical forced-token unit compatibility.
        if hasattr(d,"get_allowed_next_token_ids"):
            allowed=list(d.get_allowed_next_token_ids())
            if len(allowed)!=1:return []
            tid=allowed[0]
            nd=d.clone();nd.consume_candidate_token(tid)
            ids=list(beam.token_ids)+[tid]
            roles=list(beam.schema_role_ids)+[beam.schema_role_ids[-1] if beam.schema_role_ids else 0]
            if nd.is_candidate_complete():nd.complete_candidate()
            child=SQLBeam(ids,roles,beam.score,nd,beam.prompt_length,beam.scored_token_count,beam.grounding_score,beam.grounding_decisions)
            child.finished=bool(getattr(nd,"finished",False))
            return [child]
        return []

    def _diversity_key(self,beam,fine=False):
        d=beam.decoder
        if d is None:return ("NONE",tuple(beam.token_ids))
        where,order,limit=self._clause_presence(d)
        key=(self._projection_family(d),self._arity(d),getattr(d.grammar,"state",None),getattr(d,"selected_table",None),where,order,limit)
        if fine:key+=(tuple(getattr(d,"projection_columns",[])),getattr(d,"where_column",None),getattr(d,"order_column",None),getattr(d,"selected_operator",None),getattr(d,"selected_direction",None))
        return key

    def _select_active_beams(self,candidates):
        ranked=sorted(
            candidates,key=self.selection_score,reverse=True
        )
        if len(ranked)<=self.beam_width:
            return ranked

        selected=[]

        # First preserve the highest-scoring hypothesis for each live SQL
        # grammar state. This is purely syntactic beam diversity and prevents
        # valid LIMIT/ORDER/WHERE continuations from being pruned before the
        # model can score their completed SQL.
        seen_states=set()
        for beam in ranked:
            state=getattr(
                getattr(beam.decoder,"grammar",None),
                "state",
                None,
            )
            table=getattr(
                beam.decoder,
                "selected_table",
                None,
            )
            state_key=(state,table)
            if state_key in seen_states:
                continue
            seen_states.add(state_key)
            selected.append(beam)
            if len(selected)>=self.beam_width:
                return selected

        seen={self._diversity_key(b,False) for b in selected}
        coarse_budget=max(
            len(selected),
            int(self.beam_width*0.7),
        )
        for beam in ranked:
            if beam in selected:
                continue
            key=self._diversity_key(beam,False)
            if key in seen:
                continue
            seen.add(key)
            selected.append(beam)
            if len(selected)>=coarse_budget:
                break

        fine={self._diversity_key(b,True) for b in selected}
        for beam in ranked:
            if beam in selected:
                continue
            key=self._diversity_key(beam,True)
            if key in fine:
                continue
            fine.add(key)
            selected.append(beam)
            if len(selected)>=self.beam_width:
                return selected

        for beam in ranked:
            if beam not in selected:
                selected.append(beam)
            if len(selected)>=self.beam_width:
                break
        return selected

    def _beam_sql(self,beam):
        if beam.decoder is None:return "<tokens>"
        return beam.decoder.tokenizer.decode_ids(beam.token_ids[beam.prompt_length:]).strip()

    def search(self,initial_beam,max_steps=28):
        active=[initial_beam];finished=[]
        for _ in range(max_steps):
            candidates=[]
            for beam in active:candidates.extend(self.expand_beam(beam))
            if not candidates:break
            finished.extend([b for b in candidates if b.finished])
            active_candidates=[b for b in candidates if not b.finished]
            if not active_candidates:
                active=[];break
            active=self._select_active_beams(active_candidates)
        best={}
        for beam in finished:
            sql=self._beam_sql(beam)
            if sql not in best or self.selection_score(beam)>self.selection_score(best[sql]):best[sql]=beam
        return list(best.values())+active

    def executable_beams(self,beams):
        out=[]
        for beam in beams:
            if not beam.finished:continue
            sql=self._beam_sql(beam)
            beam.execution_valid=beam.decoder.execution_validator.validate_sql(sql)
            if beam.execution_valid:out.append(beam)
        return sorted(out,key=self.selection_score,reverse=True)

    def select_best_executable(self,beams):
        executable=self.executable_beams(beams)
        if not executable:raise RuntimeError("No executable SQL hypothesis found.")
        return executable[0]
