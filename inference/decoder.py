from layers.softmax import Softmax
from inference.beam_search import BeamSearch
from inference.repetition_penalty import RepetitionPenalty
from inference.sampler import Sampler
from inference.temperature import Temperature
from inference.top_k import TopKSampler
from inference.top_p import TopPSampler


class Decoder:
    """
    Handles text-generation strategies for a trained TinyGPT model.

    Supported strategies:
        greedy
        top_k
        top_p
        beam
    """

    def __init__(self, model) -> None:
        if model is None:
            raise ValueError("model cannot be None.")

        self.model = model
        self.softmax = Softmax()
        self.temperature = Temperature()
        self.sampler = Sampler()

    def generate(self,input_ids: list[int],maximum_new_tokens: int,strategy: str = "greedy",eos_token_id: int | None = None,
                 temperature: float = 1.0,top_k: int = 40,top_p: float = 0.9,repetition_penalty: float | None = None,beam_width: int = 3) -> list[int]:
        self._validate_generation_inputs(input_ids=input_ids,maximum_new_tokens=maximum_new_tokens,strategy=strategy,eos_token_id=eos_token_id)

        if strategy == "greedy":
            return self.generate_greedy(input_ids=input_ids,maximum_new_tokens=maximum_new_tokens,eos_token_id=eos_token_id,repetition_penalty=repetition_penalty)

        if strategy == "top_k":
            return self.generate_top_k(input_ids=input_ids,maximum_new_tokens=maximum_new_tokens,top_k=top_k,
                                       temperature=temperature,eos_token_id=eos_token_id,repetition_penalty=repetition_penalty)

        if strategy == "top_p":
            return self.generate_top_p(input_ids=input_ids,maximum_new_tokens=maximum_new_tokens,top_p=top_p,
                                       temperature=temperature,eos_token_id=eos_token_id,repetition_penalty=repetition_penalty)

        if strategy == "beam":
            return self.generate_beam_search(input_ids=input_ids,maximum_new_tokens=maximum_new_tokens,
                                             beam_width=beam_width,eos_token_id=eos_token_id)

        raise ValueError("strategy must be one of: 'greedy', 'top_k', 'top_p', or 'beam'.")

    def generate_greedy(self,input_ids: list[int],maximum_new_tokens: int,eos_token_id: int | None = None,
                        repetition_penalty: float | None = None) -> list[int]:
        generated_ids = input_ids.copy()
        penalty_handler = self._create_repetition_penalty(repetition_penalty)

        for _ in range(maximum_new_tokens):
            if self._context_limit_reached(generated_ids):
                break

            logits = self.model.forward(generated_ids)
            last_logits = logits[-1].copy()

            if penalty_handler is not None:
                last_logits = penalty_handler.forward(logits=last_logits,generated_ids=generated_ids)
            next_token_id = self._argmax(last_logits)
            generated_ids.append(next_token_id)
            if self._is_eos(next_token_id, eos_token_id):
                break
        return generated_ids

    def generate_top_k(self,input_ids: list[int],maximum_new_tokens: int,top_k: int,temperature: float = 1.0,
                       eos_token_id: int | None = None,repetition_penalty: float | None = None) -> list[int]:
        self._validate_temperature(temperature)
        if not isinstance(top_k, int):
            raise TypeError("top_k must be an integer.")
        if top_k <= 0:
            raise ValueError("top_k must be positive.")
        safe_top_k = min(top_k,self.model.vocabulary_size)
        top_k_sampler = TopKSampler(k=safe_top_k)
        penalty_handler = self._create_repetition_penalty(repetition_penalty)
        generated_ids = input_ids.copy()
        for _ in range(maximum_new_tokens):
            if self._context_limit_reached(generated_ids):
                break
            logits = self.model.forward(generated_ids)
            last_logits = logits[-1].copy()
            if penalty_handler is not None:
                last_logits = penalty_handler.forward(logits=last_logits,generated_ids=generated_ids)
            adjusted_logits = self.temperature.forward(logits=last_logits,temperature=temperature)
            top_indices, top_logits = (
                top_k_sampler.get_top_k(adjusted_logits))
            probabilities = self.softmax.forward(top_logits)
            selected_position = self.sampler.sample(probabilities)
            next_token_id = top_indices[selected_position]
            generated_ids.append(next_token_id)
            if self._is_eos(next_token_id, eos_token_id):
                break
        return generated_ids

    def generate_top_p(self,input_ids: list[int],maximum_new_tokens: int,top_p: float,temperature: float = 1.0,
                       eos_token_id: int | None = None,repetition_penalty: float | None = None) -> list[int]:
        self._validate_temperature(temperature)
        top_p_sampler = TopPSampler(p=top_p)
        penalty_handler = self._create_repetition_penalty(repetition_penalty)
        generated_ids = input_ids.copy()
        for _ in range(maximum_new_tokens):
            if self._context_limit_reached(generated_ids):
                break
            logits = self.model.forward(generated_ids)
            last_logits = logits[-1].copy()
            if penalty_handler is not None:
                last_logits = penalty_handler.forward(logits=last_logits,generated_ids=generated_ids)
            adjusted_logits = self.temperature.forward(logits=last_logits,temperature=temperature)
            probabilities = self.softmax.forward(adjusted_logits)
            sorted_indices, sorted_probabilities = self._sort_probabilities_descending(probabilities)
            selected_indices,selected_probabilities = top_p_sampler.get_top_p(indices=sorted_indices,probabilities=sorted_probabilities)
            normalized_probabilities = (self._normalize_probabilities(selected_probabilities))
            selected_position = self.sampler.sample(normalized_probabilities)
            next_token_id = selected_indices[selected_position]
            generated_ids.append(next_token_id)
            if self._is_eos(next_token_id, eos_token_id):
                break
        return generated_ids

    def generate_beam_search(self,input_ids: list[int],maximum_new_tokens: int,beam_width: int,eos_token_id: int | None = None) -> list[int]:
        if not isinstance(beam_width, int):
            raise TypeError("beam_width must be an integer.")

        if beam_width <= 0:
            raise ValueError("beam_width must be positive.")
        safe_beam_width = min(beam_width,self.model.vocabulary_size,)
        beam_search = BeamSearch(beam_width=safe_beam_width)
        beam_top_k_sampler = TopKSampler(k=safe_beam_width)
        return beam_search.generate(model=self.model,input_ids=input_ids,maximum_new_tokens=maximum_new_tokens,
                                    softmax=self.softmax,top_k_sampler=beam_top_k_sampler,eos_token_id=eos_token_id)

    def _sort_probabilities_descending(self,probabilities: list[float]) -> tuple[list[int], list[float]]:
        indexed_probabilities = [(index, probability) for index, probability in enumerate(probabilities)]
        indexed_probabilities.sort(key=lambda item: item[1],reverse=True)
        sorted_indices = [index for index, _ in indexed_probabilities]
        sorted_probabilities = [probability for _, probability in indexed_probabilities]
        return sorted_indices,sorted_probabilities,

    def _normalize_probabilities(self,probabilities: list[float]) -> list[float]:
        if not probabilities:
            raise ValueError("probabilities cannot be empty.")
        total_probability = sum(probabilities)
        if total_probability <= 0.0:
            raise ValueError("Probability total must be positive.")

        return [probability / total_probability for probability in probabilities]

    def _create_repetition_penalty(self,repetition_penalty: float | None) -> RepetitionPenalty | None:
        if repetition_penalty is None:
            return None

        return RepetitionPenalty(penalty=repetition_penalty)

    def _argmax(self,values: list[float]) -> int:
        if not values:
            raise ValueError("values cannot be empty.")

        best_index = 0

        for index in range(1, len(values)):
            if values[index] > values[best_index]:
                best_index = index

        return best_index

    def _context_limit_reached(self,generated_ids: list[int]) -> bool:
        return len(generated_ids) >= self.model.maximum_sequence_length

    def _is_eos(self,token_id: int,eos_token_id: int | None) -> bool:
        return eos_token_id is not None and token_id == eos_token_id

    def _validate_temperature(self,temperature: float) -> None:
        if not isinstance(temperature,(int, float)):
            raise TypeError("temperature must be numeric.")

        if temperature <= 0.0:
            raise ValueError("temperature must be positive.")

    def _validate_generation_inputs(self,input_ids: list[int],maximum_new_tokens: int,strategy: str,eos_token_id: int | None) -> None:
        if not isinstance(input_ids, list):
            raise TypeError("input_ids must be a list.")

        if not input_ids:
            raise ValueError("input_ids cannot be empty.")

        if any(not isinstance(token_id, int) for token_id in input_ids):
            raise TypeError("Every input token ID must be an integer.")

        for token_id in input_ids:
            if token_id < 0 or token_id >= self.model.vocabulary_size:
                raise ValueError(f"Token ID {token_id} is outside the vocabulary range.")

        if len(input_ids) > self.model.maximum_sequence_length:
            raise ValueError("Input sequence exceeds the model's maximum sequence length.")

        if not isinstance(maximum_new_tokens, int):
            raise TypeError("maximum_new_tokens must be an integer.")

        if maximum_new_tokens <= 0:
            raise ValueError("maximum_new_tokens must be positive.")

        if not isinstance(strategy, str):
            raise TypeError("strategy must be a string.")

        if eos_token_id is not None:
            if not isinstance(eos_token_id, int):
                raise TypeError("eos_token_id must be an integer or None.")
            if eos_token_id < 0 or eos_token_id >= self.model.vocabulary_size:
                raise ValueError("eos_token_id is outside the vocabulary range.")