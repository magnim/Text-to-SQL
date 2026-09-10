
import math
from types import SimpleNamespace

import torch

from text_to_sql.sql_beam import SQLBeam, SQLBeamSearch


class DummySearch(SQLBeamSearch):
    def __init__(self):
        super().__init__(model=None, decoder=None, beam_width=1)
        self.calls = 0

    def expand_beam(self, beam):
        self.calls += 1
        if self.calls == 1:
            finished = SQLBeam([1], [0], -2.0, None, 0)
            finished.finished = True
            active = SQLBeam([2], [0], -0.1, None, 0)
            return [finished, active]
        return []


def test_finished_beam_survives_active_beam_pruning():
    initial = SQLBeam([], [], 0.0, None, 0)
    beams = DummySearch().search(initial, max_steps=2)
    assert any(beam.finished for beam in beams)


class FakeDecoder:
    def __init__(self):
        self.active_token_type = "DISTINCT"
        self.active_candidates = {"DISTINCT ": [1, 2]}
        self.generated_candidate_ids = [1]
        self.grammar = SimpleNamespace(state=-1)
        self.table_candidates = []
        self.value_candidates = []
        self.finished = False

    def clone(self):
        copy = FakeDecoder()
        copy.active_token_type = self.active_token_type
        copy.active_candidates = {k: v[:] for k, v in self.active_candidates.items()}
        copy.generated_candidate_ids = self.generated_candidate_ids[:]
        return copy

    def get_allowed_next_token_ids(self):
        return [2]

    def consume_candidate_token(self, token_id):
        self.generated_candidate_ids.append(token_id)

    def is_candidate_complete(self):
        return self.generated_candidate_ids == [1, 2]

    def complete_candidate(self):
        self.active_token_type = None
        self.active_candidates = {}
        self.generated_candidate_ids = []


class FakeModel:
    def forward(self, token_ids, role_ids):
        # token 2 is legal but intentionally low-probability relative to the vocabulary.
        return torch.tensor([[[8.0, 7.0, -8.0]]], dtype=torch.float32)


def test_forced_fixed_multi_token_continuation_has_no_extra_penalty():
    decoder = FakeDecoder()
    beam = SQLBeam([99, 1], [0, 0], -1.5, decoder, 1)
    search = SQLBeamSearch(FakeModel(), decoder, beam_width=3)
    children = search.expand_beam(beam)
    assert len(children) == 1
    assert children[0].score == beam.score


def _operator_beam(operator):
    decoder = SimpleNamespace(selected_operator=operator)
    return SQLBeam([1], [0], -1.0, decoder, 0)


def test_operator_head_ranks_specific_where_operator_not_generic_where():
    # NONE, >, <, =, IN
    log_probs = [-8.0, -5.0, -6.0, -0.1, -7.0]
    search = SQLBeamSearch(
        model=None,
        decoder=None,
        beam_width=3,
        operator_log_probabilities=log_probs,
        operator_weight=1.0,
    )
    eq_score = search.selection_score(_operator_beam("="))
    less_score = search.selection_score(_operator_beam("<"))
    assert eq_score > less_score


class PrefixTokenizer:
    def __init__(self, text):
        self.text = text

    def decode_ids(self, ids):
        return self.text


def test_active_projection_family_respects_established_avg_prefix():
    decoder = SimpleNamespace(
        projection_family=None,
        active_token_type="COLUMN",
        tokenizer=PrefixTokenizer("SELECT AVG("),
    )
    beam = SQLBeam([1], [0], -1.0, decoder, 0)
    search = SQLBeamSearch(model=None, decoder=None, beam_width=3)

    assert search._active_projection_family(beam) == "AVG"
