from src.embeddings.embedding import Embedding
from src.embeddings.positional_embed import PositionalEmbedding
from layers.layer_normalization import LayerNormalization
from layers.linear import Linear
from layers.transformer_block import TransformerBlock

from inference.temperature import Temperature
from inference.top_k import TopKSampler
from inference.sampler import Sampler
from layers.softmax import Softmax


class TinyGPT:
    def __init__(self,vocabulary_size: int,embedding_dimension: int,maximum_sequence_length: int,number_of_heads: int,
                 hidden_dimension: int,number_of_layers: int) -> None:
        self._validate_configuration(vocabulary_size,embedding_dimension,maximum_sequence_length,number_of_heads,hidden_dimension,number_of_layers)

        self.vocabulary_size = vocabulary_size
        self.embedding_dimension = embedding_dimension
        self.maximum_sequence_length = maximum_sequence_length
        self.number_of_heads = number_of_heads
        self.hidden_dimension = hidden_dimension
        self.number_of_layers = number_of_layers

        self.token_embedding = Embedding(vocabulary_size,embedding_dimension)

        self.position_embedding = PositionalEmbedding(maximum_sequence_length,embedding_dimension)

        self.transformer_blocks = [TransformerBlock(embedding_dimension,number_of_heads,hidden_dimension) for _ in range(number_of_layers)]

        self.final_layer_norm = LayerNormalization(embedding_dimension)

        self.output_projection = Linear(embedding_dimension,vocabulary_size)

        self.last_input_ids: list[int] | None = None
        self.last_token_embeddings: list[list[float]] | None = None
        self.last_position_embeddings: list[list[float]] | None = None
        self.last_combined_embeddings: list[list[float]] | None = None
        self.last_transformer_output: list[list[float]] | None = None
        self.last_normalized_output: list[list[float]] | None = None
        self.last_logits: list[list[float]] | None = None

        self.temperature = Temperature()
        self.top_k_sampler = TopKSampler(k=min(40, vocabulary_size))
        self.softmax = Softmax()
        self.sampler = Sampler()

    def forward(self, input_ids: list[int]) -> list[list[float]]:
        self._validate_input_ids(input_ids)
        self.last_input_ids = input_ids.copy()

        token_embeddings = self.token_embedding.forward(input_ids)
        self.last_token_embeddings = [row.copy() for row in token_embeddings]

        position_embeddings = self.position_embedding.forward(len(input_ids))
        self.last_position_embeddings = [row.copy() for row in position_embeddings]

        hidden_states = self._add_embeddings(token_embeddings,position_embeddings)
        self.last_combined_embeddings = [row.copy() for row in hidden_states]

        for transformer_block in self.transformer_blocks:
            hidden_states = transformer_block.forward(hidden_states)

        self.last_transformer_output = [row.copy() for row in hidden_states]

        hidden_states = self.final_layer_norm.forward(hidden_states)
        self.last_normalized_output = [row.copy() for row in hidden_states]

        logits = self.output_projection.forward(hidden_states)
        self.last_logits = [row.copy() for row in logits]

        return logits

    def backward(self, logits_gradients: list[list[float]]) -> list[list[float]]:
        self._validate_logits_gradients(logits_gradients)

        hidden_gradients = self.output_projection.backward(logits_gradients)
        hidden_gradients = self.final_layer_norm.backward(hidden_gradients)

        for transformer_block in reversed(self.transformer_blocks):
            hidden_gradients = transformer_block.backward(hidden_gradients)

        token_embedding_gradients = [row.copy() for row in hidden_gradients]

        position_embedding_gradients = [row.copy() for row in hidden_gradients]

        self.token_embedding.backward(token_embedding_gradients)
        self.position_embedding.backward(position_embedding_gradients)

        return hidden_gradients

    def _add_embeddings(self,token_embeddings: list[list[float]],position_embeddings: list[list[float]]) -> list[list[float]]:
        if len(token_embeddings) != len(position_embeddings):
            raise ValueError("Token and positional embeddings must have the same sequence length.")

        combined_embeddings = []

        for token_row, position_row in zip(token_embeddings,position_embeddings):
            if len(token_row) != self.embedding_dimension:
                raise ValueError(f"Token embedding rows must contain {self.embedding_dimension} values.")

            if len(position_row) != self.embedding_dimension:
                raise ValueError(f"Position embedding rows must contain {self.embedding_dimension} values.")

            combined_embeddings.append([token_value + position_value for token_value, position_value in zip(token_row,position_row)])

        return combined_embeddings

    def _validate_input_ids(self, input_ids: list[int]) -> None:
        if not isinstance(input_ids, list):
            raise TypeError("input_ids must be a list.")
        if not input_ids:
            raise ValueError("input_ids cannot be empty.")
        if len(input_ids) > self.maximum_sequence_length:
            raise ValueError("Input sequence length exceeds maximum_sequence_length.")

        for token_id in input_ids:
            if not isinstance(token_id, int):
                raise TypeError("Every token ID must be an integer.")
            if token_id < 0 or token_id >= self.vocabulary_size:
                raise ValueError(f"Token ID {token_id} is outside the vocabulary range.")

    def _validate_logits_gradients(self,logits_gradients: list[list[float]]) -> None:
        if self.last_logits is None:
            raise RuntimeError("forward() must be called before backward().")
        if not isinstance(logits_gradients, list):
            raise TypeError("logits_gradients must be a list.")
        if not logits_gradients:
            raise ValueError("logits_gradients cannot be empty.")
        if len(logits_gradients) != len(self.last_logits):
            raise ValueError(
                "Gradient sequence length must match the logits sequence length."
            )

        for gradient_row in logits_gradients:
            if not isinstance(gradient_row, list):
                raise TypeError("Every gradient row must be a list.")
            if len(gradient_row) != self.vocabulary_size:
                raise ValueError(
                    f"Every gradient row must contain {self.vocabulary_size} values."
                )
            if any(
                not isinstance(gradient, (int, float))
                for gradient in gradient_row
            ):
                raise TypeError("Every logit gradient must be numeric.")

    def _validate_configuration(self,vocabulary_size: int,embedding_dimension: int,maximum_sequence_length: int,
                                number_of_heads: int,hidden_dimension: int,number_of_layers: int) -> None:
        configuration_values = {
            "vocabulary_size": vocabulary_size,
            "embedding_dimension": embedding_dimension,
            "maximum_sequence_length": maximum_sequence_length,
            "number_of_heads": number_of_heads,
            "hidden_dimension": hidden_dimension,
            "number_of_layers": number_of_layers,
        }

        for name, value in configuration_values.items():
            if not isinstance(value, int):
                raise TypeError(f"{name} must be an integer.")
        if vocabulary_size <= 0:
            raise ValueError("vocabulary_size must be positive.")
        if embedding_dimension <= 0:
            raise ValueError("embedding_dimension must be positive.")
        if maximum_sequence_length <= 0:
            raise ValueError("maximum_sequence_length must be positive.")
        if number_of_heads <= 0:
            raise ValueError("number_of_heads must be positive.")
        if hidden_dimension <= 0:
            raise ValueError("hidden_dimension must be positive.")
        if number_of_layers <= 0:
            raise ValueError("number_of_layers must be positive.")
        if embedding_dimension % number_of_heads != 0:
            raise ValueError("embedding_dimension must be divisible by number_of_heads.")

    def update_parameters(self, learning_rate: float) -> None:
        if learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive.")

        self.token_embedding.update_parameters(learning_rate)
        self.position_embedding.update_parameters(learning_rate)

        for transformer_block in self.transformer_blocks:
            transformer_block.update_parameters(learning_rate)

        self.final_layer_norm.update_parameters(learning_rate)
        self.output_projection.update_parameters(learning_rate)

    def _argmax(self,values: list[float]) -> int:

        if not values:
            raise ValueError("values cannot be empty.")
        if not isinstance(values, list):
            raise TypeError("values must be a list.")
        if any(not isinstance(value, (int, float)) for value in values):
            raise TypeError("Every value must be numeric.")
        best_index = 0
        for index in range(1,len(values)):
            if values[index] > values[best_index]:
                best_index = index
        return best_index

    def generate(self,input_ids: list[int],maximum_new_tokens: int,eos_token_id: int | None = None) -> list[int]:
        self._validate_input_ids(input_ids)
        if not isinstance(maximum_new_tokens, int):
            raise TypeError("maximum_new_tokens must be an integer.")

        if maximum_new_tokens <= 0:
            raise ValueError("maximum_new_tokens must be positive.")

        if eos_token_id is not None:
            if not isinstance(eos_token_id, int):
                raise TypeError("eos_token_id must be an integer or None.")

            if eos_token_id < 0 or eos_token_id >= self.vocabulary_size:
                raise ValueError("eos_token_id is outside the vocabulary range.")
        generated_ids = input_ids.copy()
        for _ in range(maximum_new_tokens):
            if len(generated_ids) >= self.maximum_sequence_length:
                break
            logits = self.forward(generated_ids)
            last_logits = logits[-1]
            adjusted_logits = self.temperature.forward(last_logits,temperature=1.0)
            top_indices, top_logits = (self.top_k_sampler.get_top_k(adjusted_logits))
            probabilities = self.softmax.forward(top_logits)
            selected_position = (self.sampler.sample(probabilities))
            next_token = top_indices[selected_position]
            generated_ids.append(next_token)
            if eos_token_id is not None and next_token == eos_token_id:
                break

        return generated_ids



