import math

class Beam:
    def __init__(self,sequence: list[int],score: float) -> None:
        if not isinstance(sequence, list):
            raise TypeError("sequence must be a list.")
        if not sequence:
            raise ValueError("sequence cannot be empty.")
        if any(not isinstance(token_id, int) for token_id in sequence):
            raise TypeError("Every token ID must be an integer.")
        if not isinstance(score, (int, float)):
            raise TypeError("score must be numeric.")
        self.sequence = sequence.copy()
        self.score = float(score)

    def extend(self,token_id: int,token_log_probability: float) -> "Beam":
        if not isinstance(token_id, int):
            raise TypeError("token_id must be an integer.")
        if not isinstance(token_log_probability,(int, float)):
            raise TypeError("token_log_probability must be numeric.")
        new_sequence = self.sequence + [token_id]
        new_score = self.score + token_log_probability
        return Beam(sequence=new_sequence,score=new_score)


class BeamSearch:
    def __init__(self,beam_width: int):
        if beam_width <= 0:
            raise ValueError("beam_width must be positive.")
        self.beam_width = beam_width

    def _expand_beam(self,beam: Beam,token_ids: list[int],probabilities: list[float]) -> list[Beam]:
        if len(token_ids) != len(probabilities):
            raise ValueError("token_ids and probabilities must have the same length.")
        expanded_beams = []
        for token_id, probability in zip(token_ids,probabilities):
            token_log_probability = math.log(max(probability, 1e-12))
            new_beam = beam.extend(token_id=token_id,token_log_probability=token_log_probability)
            expanded_beams.append(new_beam)
        return expanded_beams

    def _keep_best_beams(self,beams: list[Beam]) -> list[Beam]:
        if not beams:
            raise ValueError("beams cannot be empty.")
        working_beams = beams.copy()
        best_beams = []
        beam_count = min(self.beam_width,len(working_beams))

        for _ in range(beam_count):
            best_index = 0

            for index in range(1,len(working_beams)):
                if working_beams[index].score>working_beams[best_index].score:
                    best_index = index
            best_beams.append(working_beams[best_index])
            working_beams.pop(best_index)
        return best_beams

    def generate(self,model,input_ids: list[int],maximum_new_tokens: int,softmax,top_k_sampler,eos_token_id: int | None = None) -> list[int]:
        if not isinstance(input_ids, list):
            raise TypeError("input_ids must be a list.")
        if not input_ids:
            raise ValueError("input_ids cannot be empty.")
        if maximum_new_tokens <= 0:
            raise ValueError("maximum_new_tokens must be positive.")
        beams = [Beam(sequence=input_ids,score=0.0)]

        for _ in range(maximum_new_tokens):
            candidate_beams = []
            for beam in beams:
                if eos_token_id is not None and beam.sequence[-1] == eos_token_id:
                    candidate_beams.append(beam)
                    continue

                if len(beam.sequence) >= model.maximum_sequence_length:
                    candidate_beams.append(beam)
                    continue
                logits = model.forward(beam.sequence)
                last_logits = logits[-1]
                probabilities = softmax.forward(last_logits)
                (
                    top_token_ids,
                    top_probabilities,
                ) = top_k_sampler.get_top_k(probabilities)
                expanded_beams = self._expand_beam(beam=beam,token_ids=top_token_ids,probabilities=top_probabilities)
                candidate_beams.extend(expanded_beams)
            if not candidate_beams:
                break
            beams = self._keep_best_beams(candidate_beams)
            all_beams_finished = all((eos_token_id is not None and beam.sequence[-1] == eos_token_id)
                                     or (len(beam.sequence)>= model.maximum_sequence_length)for beam in beams)
            if all_beams_finished:
                break
        best_beam = self._keep_best_beams(beams)[0]

        return best_beam.sequence