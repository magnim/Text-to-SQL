import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module):
    def __init__(self, embedding_dimension: int, hidden_dimension: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(embedding_dimension, hidden_dimension),
            nn.GELU(),
            nn.Linear(hidden_dimension, embedding_dimension),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class TransformerBlock(nn.Module):
    def __init__(self, embedding_dimension: int, number_of_heads: int, hidden_dimension: int) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embedding_dimension, number_of_heads, batch_first=True
        )
        self.norm1 = nn.LayerNorm(embedding_dimension)
        self.feed_forward = FeedForward(embedding_dimension, hidden_dimension)
        self.norm2 = nn.LayerNorm(embedding_dimension)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = x.size(1)
        causal_mask = torch.triu(
            torch.ones(n, n, device=x.device, dtype=torch.bool), diagonal=1
        )
        attn, _ = self.attention(x, x, x, attn_mask=causal_mask, need_weights=False)
        x = self.norm1(x + attn)
        return self.norm2(x + self.feed_forward(x))


class TinyGPT(nn.Module):
    """Decoder-only Transformer with learned schema-aware semantic heads."""

    def __init__(
        self,
        vocabulary_size: int,
        embedding_dimension: int,
        maximum_sequence_length: int,
        number_of_heads: int,
        hidden_dimension: int,
        number_of_layers: int,
        number_of_schema_roles: int = 5,
    ) -> None:
        super().__init__()
        self.vocabulary_size = vocabulary_size
        self.embedding_dimension = embedding_dimension
        self.maximum_sequence_length = maximum_sequence_length
        self.number_of_heads = number_of_heads
        self.hidden_dimension = hidden_dimension
        self.number_of_layers = number_of_layers
        self.number_of_schema_roles = number_of_schema_roles

        self.token_embedding = nn.Embedding(vocabulary_size, embedding_dimension)
        self.position_embedding = nn.Embedding(maximum_sequence_length, embedding_dimension)
        self.schema_role_embedding = nn.Embedding(number_of_schema_roles, embedding_dimension)
        self.blocks = nn.ModuleList([
            TransformerBlock(embedding_dimension, number_of_heads, hidden_dimension)
            for _ in range(number_of_layers)
        ])
        self.final_norm = nn.LayerNorm(embedding_dimension)
        self.output_projection = nn.Linear(embedding_dimension, vocabulary_size, bias=False)
        self.output_projection.weight = self.token_embedding.weight

        # Contextual pointer projections at the active SQL generation position.
        self.column_query_projection = nn.Linear(embedding_dimension, embedding_dimension, bias=False)
        self.column_key_projection = nn.Linear(embedding_dimension, embedding_dimension, bias=False)
        self.table_query_projection = nn.Linear(embedding_dimension, embedding_dimension, bias=False)
        self.table_key_projection = nn.Linear(embedding_dimension, embedding_dimension, bias=False)

        # Question pooling and structural intent heads.
        self.question_pool_score = nn.Linear(embedding_dimension, 1, bias=False)
        self.projection_head = nn.Linear(embedding_dimension, 5)   # STAR/COLUMN/DISTINCT/COUNT/AVG
        self.projection_arity_head = nn.Linear(embedding_dimension, 3)  # zero/one/multi
        self.clause_head = nn.Linear(embedding_dimension, 3)       # WHERE/ORDER/LIMIT
        self.operator_head = nn.Linear(embedding_dimension, 5)     # NONE/>/</=/IN
        self.direction_head = nn.Linear(embedding_dimension, 3)    # NONE/ASC/DESC
        self.continuation_head = nn.Linear(embedding_dimension, 4) # NONE/ORDER/LIMIT/ORDER+LIMIT

        # A separately pooled semantic structure ensemble improves paraphrase
        # transfer without changing the Transformer backbone. It consumes the
        # same full-schema-conditioned QUESTION hidden states, but uses simple
        # mean pooling instead of the legacy-preservation attention pool.
        self.semantic_clause_head = nn.Sequential(
            nn.Linear(embedding_dimension, embedding_dimension),
            nn.GELU(),
            nn.Linear(embedding_dimension, 3),
        )
        self.semantic_continuation_head = nn.Sequential(
            nn.Linear(embedding_dimension, embedding_dimension),
            nn.GELU(),
            nn.Linear(embedding_dimension, 4),
        )

        # Schema-invariant learned structure heads over QUESTION token
        # embeddings. These separate SQL clause intent from schema identity,
        # while all identifier selection remains full-schema-conditioned.
        self.lexical_question_pool_score = nn.Linear(
            embedding_dimension, 1, bias=False
        )
        self.lexical_clause_head = nn.Sequential(
            nn.Linear(embedding_dimension, embedding_dimension),
            nn.GELU(),
            nn.Linear(embedding_dimension, 3),
        )
        self.lexical_continuation_head = nn.Sequential(
            nn.Linear(embedding_dimension, embedding_dimension),
            nn.GELU(),
            nn.Linear(embedding_dimension, 4),
        )

        # Order-invariant semantic table/column grounding from QUESTION states.
        self.lexical_table_query_projection = nn.Linear(embedding_dimension, embedding_dimension, bias=False)
        self.lexical_table_key_projection = nn.Linear(embedding_dimension, embedding_dimension, bias=False)
        self.lexical_column_query_projection = nn.Linear(embedding_dimension, embedding_dimension, bias=False)
        self.lexical_column_key_projection = nn.Linear(embedding_dimension, embedding_dimension, bias=False)

        # Clause-specific token-attention grounding for compositions.
        self.where_question_pool_score = nn.Linear(embedding_dimension, 1, bias=False)
        self.order_question_pool_score = nn.Linear(embedding_dimension, 1, bias=False)
        self.where_column_query_projection = nn.Linear(embedding_dimension, embedding_dimension, bias=False)
        self.where_column_key_projection = nn.Linear(embedding_dimension, embedding_dimension, bias=False)
        self.order_column_query_projection = nn.Linear(embedding_dimension, embedding_dimension, bias=False)
        self.order_column_key_projection = nn.Linear(embedding_dimension, embedding_dimension, bias=False)

        # Learned role-specific numeric pointers over QUESTION token states.
        # They distinguish comparison values from LIMIT values without any
        # phrase-to-number rule.
        self.where_value_score = nn.Linear(
            embedding_dimension, 1, bias=False
        )
        self.limit_value_score = nn.Linear(
            embedding_dimension, 1, bias=False
        )

        # Case-normalized identifier auxiliary signal for schemas such as Age/ID.
        self.normalized_column_query_projection = nn.Linear(embedding_dimension, embedding_dimension, bias=False)
        self.normalized_column_key_projection = nn.Linear(embedding_dimension, embedding_dimension, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _normalize_inputs(self, input_ids, schema_role_ids=None):
        if isinstance(input_ids, list):
            input_ids = torch.tensor(input_ids, dtype=torch.long)
        if isinstance(schema_role_ids, list):
            schema_role_ids = torch.tensor(schema_role_ids, dtype=torch.long)
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        if schema_role_ids is None:
            schema_role_ids = torch.zeros_like(input_ids)
        elif schema_role_ids.dim() == 1:
            schema_role_ids = schema_role_ids.unsqueeze(0)
        return input_ids, schema_role_ids

    def _get_hidden_states(self, input_ids, schema_role_ids=None):
        input_ids, schema_role_ids = self._normalize_inputs(input_ids, schema_role_ids)
        batch_size, sequence_length = input_ids.shape
        if sequence_length > self.maximum_sequence_length:
            raise ValueError("Input sequence length exceeds maximum_sequence_length.")
        positions = torch.arange(sequence_length, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        x = (
            self.token_embedding(input_ids)
            + self.position_embedding(positions)
            + self.schema_role_embedding(schema_role_ids)
        )
        for block in self.blocks:
            x = block(x)
        return self.final_norm(x)

    def forward(self, input_ids, schema_role_ids=None):
        return self.output_projection(self._get_hidden_states(input_ids, schema_role_ids))

    def get_hidden_states(self, input_ids, schema_role_ids=None):
        return self._get_hidden_states(input_ids, schema_role_ids)

    def config_dict(self):
        return {
            "vocabulary_size": self.vocabulary_size,
            "embedding_dimension": self.embedding_dimension,
            "maximum_sequence_length": self.maximum_sequence_length,
            "number_of_heads": self.number_of_heads,
            "hidden_dimension": self.hidden_dimension,
            "number_of_layers": self.number_of_layers,
            "number_of_schema_roles": self.number_of_schema_roles,
        }

    def question_summary_from_hidden(self, hidden_sequence, question_span):
        if hidden_sequence.dim() == 3:
            hidden_sequence = hidden_sequence[0]
        start, end = question_span
        states = hidden_sequence[start:end]
        if states.numel() == 0:
            raise ValueError("question_span cannot be empty.")
        weights = torch.softmax(self.question_pool_score(states).squeeze(-1), dim=0)
        return (states * weights.unsqueeze(-1)).sum(dim=0)

    @staticmethod
    def mean_question_state(hidden_sequence, question_span):
        if hidden_sequence.dim() == 3:
            hidden_sequence = hidden_sequence[0]
        start, end = question_span
        states = hidden_sequence[start:end]
        if states.numel() == 0:
            raise ValueError("question_span cannot be empty.")
        return states.mean(dim=0)

    def predict_intents_from_hidden(self, hidden_sequence, question_span):
        summary = self.question_summary_from_hidden(hidden_sequence, question_span)
        return {
            "projection_logits": self.projection_head(summary),
            "projection_arity_logits": self.projection_arity_head(summary),
            "clause_logits": self.clause_head(summary),
            "operator_logits": self.operator_head(summary),
            "direction_logits": self.direction_head(summary),
            "continuation_logits": self.continuation_head(summary),
        }

    def predict_lexical_structure(self, question_token_ids):
        if isinstance(question_token_ids, list):
            question_token_ids = torch.tensor(
                question_token_ids,
                dtype=torch.long,
                device=self.token_embedding.weight.device,
            )
        if question_token_ids.numel() == 0:
            raise ValueError("question_token_ids cannot be empty.")
        states = self.token_embedding(question_token_ids)
        weights = torch.softmax(
            self.lexical_question_pool_score(states).squeeze(-1),
            dim=0,
        )
        summary = (states * weights.unsqueeze(-1)).sum(dim=0)
        return {
            "clause_logits": self.lexical_clause_head(summary),
            "continuation_logits": self.lexical_continuation_head(summary),
        }

    def predict_semantic_structure_from_hidden(
        self, hidden_sequence, question_span
    ):
        summary = self.mean_question_state(
            hidden_sequence, question_span
        )
        return {
            "clause_logits": self.semantic_clause_head(summary),
            "continuation_logits": self.semantic_continuation_head(summary),
        }

    def predict_operator(self, input_ids, schema_role_ids=None):
        hidden = self.get_hidden_states(input_ids, schema_role_ids)
        return self.operator_head(hidden[:, -1, :])

    def _pointer_pair(self, kind):
        if kind == "table":
            return self.table_query_projection, self.table_key_projection
        if kind == "column":
            return self.column_query_projection, self.column_key_projection
        raise ValueError("kind must be table or column")

    def score_schema_spans_from_hidden(
        self, hidden_sequence, query_position, candidate_span_lists, kind
    ):
        if hidden_sequence.dim() == 3:
            hidden_sequence = hidden_sequence[0]
        qproj, kproj = self._pointer_pair(kind)
        query = qproj(hidden_sequence[query_position])
        scores = []
        for spans in candidate_span_lists:
            fields = []
            for start, end in spans:
                states = hidden_sequence[start:end]
                if states.numel() == 0:
                    continue
                key = kproj(states.mean(dim=0))
                fields.append((query * key).sum() / math.sqrt(self.embedding_dimension))
            scores.append(torch.logsumexp(torch.stack(fields), dim=0) if fields else query.new_tensor(-1e9))
        return torch.stack(scores)

    def _lexical_pair(self, kind):
        if kind == "table":
            return self.lexical_table_query_projection, self.lexical_table_key_projection
        if kind == "column":
            return self.lexical_column_query_projection, self.lexical_column_key_projection
        raise ValueError("kind must be table or column")

    def score_schema_fields_from_question(
        self, hidden_sequence, question_span, candidate_field_token_ids, kind
    ):
        """Order-invariant learned matching between QUESTION states and active-schema fields."""
        if hidden_sequence.dim() == 3:
            hidden_sequence = hidden_sequence[0]
        start, end = question_span
        qstates = hidden_sequence[start:end]
        qproj, kproj = self._lexical_pair(kind)
        q = qproj(qstates)
        candidate_scores = []
        for fields in candidate_field_token_ids:
            field_scores = []
            for token_ids in fields:
                if isinstance(token_ids, list):
                    token_ids = torch.tensor(
                        token_ids, dtype=torch.long, device=self.token_embedding.weight.device
                    )
                if token_ids.numel() == 0:
                    continue
                key = kproj(self.token_embedding(token_ids).mean(dim=0))
                token_scores = (q * key.unsqueeze(0)).sum(dim=-1) / math.sqrt(self.embedding_dimension)
                field_scores.append(torch.logsumexp(token_scores, dim=0))
            candidate_scores.append(
                torch.logsumexp(torch.stack(field_scores), dim=0)
                if field_scores else q.new_tensor(-1e9)
            )
        return torch.stack(candidate_scores)

    def score_clause_columns_from_question(
        self, hidden_sequence, question_span, candidate_token_id_lists, kind
    ):
        if hidden_sequence.dim() == 3:
            hidden_sequence = hidden_sequence[0]
        start, end = question_span
        states = hidden_sequence[start:end]
        if kind == "where":
            pool = self.where_question_pool_score
            qproj = self.where_column_query_projection
            kproj = self.where_column_key_projection
        elif kind == "order":
            pool = self.order_question_pool_score
            qproj = self.order_column_query_projection
            kproj = self.order_column_key_projection
        else:
            raise ValueError("kind must be where or order")
        weights = torch.softmax(pool(states).squeeze(-1), dim=0)
        query = qproj((states * weights.unsqueeze(-1)).sum(dim=0))
        scores = []
        for token_ids in candidate_token_id_lists:
            if isinstance(token_ids, list):
                token_ids = torch.tensor(
                    token_ids, dtype=torch.long, device=self.token_embedding.weight.device
                )
            if token_ids.numel() == 0:
                scores.append(query.new_tensor(-1e9))
                continue
            key = kproj(self.token_embedding(token_ids).mean(dim=0))
            scores.append((query * key).sum() / math.sqrt(self.embedding_dimension))
        return torch.stack(scores)

    def score_question_value_spans(
        self,
        hidden_sequence,
        candidate_span_lists,
        kind,
    ):
        if hidden_sequence.dim() == 3:
            hidden_sequence = hidden_sequence[0]
        if kind == "where":
            scorer = self.where_value_score
        elif kind == "limit":
            scorer = self.limit_value_score
        else:
            raise ValueError("kind must be where or limit")

        scores = []
        for spans in candidate_span_lists:
            occurrence_scores = []
            for start, end in spans:
                states = hidden_sequence[start:end]
                if states.numel() == 0:
                    continue
                occurrence_scores.append(
                    scorer(states.mean(dim=0)).squeeze(-1)
                )
            if occurrence_scores:
                scores.append(
                    torch.logsumexp(
                        torch.stack(occurrence_scores), dim=0
                    )
                )
            else:
                scores.append(
                    hidden_sequence.new_tensor(-1e9)
                )
        return torch.stack(scores)

    def score_normalized_columns_from_question(
        self, hidden_sequence, question_span, candidate_token_id_lists
    ):
        summary = self.mean_question_state(hidden_sequence, question_span)
        query = self.normalized_column_query_projection(summary)
        scores = []
        for token_ids in candidate_token_id_lists:
            if isinstance(token_ids, list):
                token_ids = torch.tensor(
                    token_ids, dtype=torch.long, device=self.token_embedding.weight.device
                )
            key = self.normalized_column_key_projection(
                self.token_embedding(token_ids).mean(dim=0)
            )
            scores.append((query * key).sum() / math.sqrt(self.embedding_dimension))
        return torch.stack(scores)

    def score_identifier_embedding_affinity(self, question_token_ids, candidate_token_ids):
        device = self.token_embedding.weight.device
        qids = torch.tensor(question_token_ids, dtype=torch.long, device=device) if isinstance(question_token_ids, list) else question_token_ids
        cids = torch.tensor(candidate_token_ids, dtype=torch.long, device=device) if isinstance(candidate_token_ids, list) else candidate_token_ids
        if qids.numel() == 0 or cids.numel() == 0:
            return self.token_embedding.weight.new_tensor(0.0)
        q = F.normalize(self.token_embedding(qids), dim=-1)
        c = F.normalize(self.token_embedding(cids), dim=-1)
        best = (q @ c.T).max(dim=0).values
        strong = torch.relu(best - 0.70)
        return strong.sum() / torch.sqrt(strong.new_tensor(float(max(strong.numel(), 1))))

    # Legacy compatibility.
    def score_column(self, input_ids, schema_role_ids, column_token_ids):
        hidden = self.get_hidden_states(input_ids, schema_role_ids)
        query = self.column_query_projection(hidden[:, -1, :])
        if isinstance(column_token_ids, list):
            column_token_ids = torch.tensor(column_token_ids, dtype=torch.long, device=query.device)
        if column_token_ids.dim() == 1:
            column_token_ids = column_token_ids.unsqueeze(0)
        key = self.column_key_projection(self.token_embedding(column_token_ids).mean(dim=1))
        return (query * key).sum(dim=-1)

    def score_table(self, input_ids, schema_role_ids, table_token_ids):
        hidden = self.get_hidden_states(input_ids, schema_role_ids)
        query = self.table_query_projection(hidden[:, -1, :])
        if isinstance(table_token_ids, list):
            table_token_ids = torch.tensor(table_token_ids, dtype=torch.long, device=query.device)
        if table_token_ids.dim() == 1:
            table_token_ids = table_token_ids.unsqueeze(0)
        key = self.table_key_projection(self.token_embedding(table_token_ids).mean(dim=1))
        return (query * key).sum(dim=-1)
