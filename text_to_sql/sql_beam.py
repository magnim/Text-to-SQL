from text_to_sql.constrained_decoder import ConstrainedDecoder
from text_to_sql.sql_grammar import SQLGrammar
import math


class SQLBeam:

    def __init__(self, token_ids: list[int], score: float, decoder: ConstrainedDecoder, prompt_length: int) -> None:
        self.token_ids = token_ids
        self.score = score
        self.finished = False
        self.decoder = decoder
        self.prompt_length = prompt_length
        self.execution_valid: bool | None = None


class SQLBeamSearch:

    def __init__(self, model, decoder: ConstrainedDecoder, beam_width: int = 3) -> None:
        if beam_width <= 0:
            raise ValueError("beam_width must be greater than 0.")

        self.model = model
        self.decoder = decoder
        self.beam_width = beam_width

    def expand_beam(self, beam: SQLBeam) -> list[SQLBeam]:
        if beam.finished:
            return [beam]
        logits = self.model.forward(beam.token_ids)
        last_logits = logits[-1]
        log_probabilities = self._log_probabilities(last_logits)
        decoder_options = []
        # Continue the currently active candidate.
        if beam.decoder.active_token_type is not None:
            decoder_options.append(beam.decoder.clone())
        # Otherwise branch across all grammar-allowed token types.
        else:
            for token_type in beam.decoder.grammar.allowed_token_types():
                decoder_copy = beam.decoder.clone()
                decoder_copy.start_candidate(token_type)
                decoder_options.append(decoder_copy)
        new_beams = []
        for decoder_copy in decoder_options:
            allowed_ids = decoder_copy.get_allowed_next_token_ids()
            ranked_ids = sorted(allowed_ids, key=lambda token_id: log_probabilities[token_id], reverse=True)
            for token_id in ranked_ids[:self.beam_width]:
                child_decoder = decoder_copy.clone()
                child_decoder.consume_candidate_token(token_id)
                new_beam = SQLBeam(token_ids=beam.token_ids + [token_id],
                                   score=beam.score + log_probabilities[token_id], decoder=child_decoder,
                                   prompt_length=beam.prompt_length)
                if child_decoder.is_candidate_complete():
                    child_decoder.complete_candidate()
                    if child_decoder.grammar.state == SQLGrammar.COMPLETE:
                        new_beam.finished = True
                new_beams.append(new_beam)
        return new_beams

    def search(self, initial_beam: SQLBeam, max_steps: int = 50) -> list[SQLBeam]:
        beams = [initial_beam]
        for _ in range(max_steps):
            candidates = []
            for beam in beams:
                candidates.extend(self.expand_beam(beam))
            if not candidates:
                break
            candidates.sort(key=lambda beam: beam.score, reverse=True)
            beams = candidates[:self.beam_width]
            if all(beam.finished for beam in beams):
                break
        return beams

    def select_best_executable(self, beams: list[SQLBeam]) -> SQLBeam:
        finished_beams = [beam for beam in beams if beam.finished]
        finished_beams.sort(key=lambda beam: beam.score, reverse=True)

        for beam in finished_beams:
            generated_ids = beam.token_ids[beam.prompt_length:]
            sql = beam.decoder.tokenizer.decode_ids(generated_ids).strip()
            print("Beam SQL:", repr(sql), "Score:", beam.score)

            beam.execution_valid = beam.decoder.execution_validator.validate_sql(sql)

            if beam.execution_valid:
                return beam

        raise RuntimeError("No executable SQL beam found.")

    def _log_probabilities(self, logits: list[float]) -> list[float]:
        maximum = max(logits)
        log_sum = maximum + math.log(sum(math.exp(logit - maximum) for logit in logits))
        return [logit - log_sum for logit in logits]
