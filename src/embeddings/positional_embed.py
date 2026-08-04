import random
from optimizers.adam import Adam


class PositionalEmbedding:

    def __init__(self,max_sequence_length,embedding_dimension):

        if max_sequence_length <= 0:
            raise ValueError("Maximum sequence length must be positive.")

        if embedding_dimension <= 0:
            raise ValueError("Embedding dimension must be positive.")

        self.max_sequence_length = max_sequence_length
        self.embedding_dimension = embedding_dimension
        self.position_matrix = [[random.uniform(-1, 1) for _ in range(embedding_dimension)]
                                for _ in range(max_sequence_length)]

        self.optimizer = Adam()
        self.position_first_moment = [[0.0 for _ in range(embedding_dimension)] for _ in range(max_sequence_length)]
        self.position_second_moment = [[0.0 for _ in range(embedding_dimension)] for _ in range(max_sequence_length)]

        self.last_positions: list[int] | None = None
        self.position_gradients: list[list[float]] | None = None

        if not isinstance(max_sequence_length, int):
            raise TypeError("max_sequence_length must be an integer.")
        if not isinstance(embedding_dimension, int):
            raise TypeError("embedding_dimension must be an integer.")

    def forward(self, sequence_length):
        if not isinstance(sequence_length, int):
            raise TypeError("sequence_length must be an integer.")
        if sequence_length <= 0:
            raise ValueError("Sequence length must be positive.")
        if sequence_length > self.max_sequence_length:
            raise ValueError("Sequence length exceeds maximum.")
        self.last_positions = list(range(sequence_length))
        return [self.position_matrix[position].copy() for position in self.last_positions]

    # no longer needed
    # def _validate_backward(self):
    #
    #     if self.last_positions is None:
    #         raise RuntimeError("forward() must be called before backward().")

    def backward(self, output_gradients: list[list[float]]) -> None:
        if self.last_positions is None:
            raise RuntimeError("positional forward() must be called before backward().")

        if not isinstance(output_gradients, list):
            raise TypeError("output_gradients must be a list.")

        if len(output_gradients) != len(self.last_positions):
            raise ValueError("positional Gradient sequence length must match the used positions.")

        for row in output_gradients:
            if not isinstance(row, list):
                raise TypeError("Every positional-gradient row must be a list.")
            if len(row) != self.embedding_dimension:
                raise ValueError(f"Each positional gradient must contain {self.embedding_dimension} values.")
            if any(not isinstance(value, (int, float)) for value in row):
                raise TypeError("Every positional-gradient value must be numeric.")

        self.position_gradients = [[0.0 for _ in range(self.embedding_dimension)] for _ in range(self.max_sequence_length)]

        for gradient_index, position in enumerate(self.last_positions):
            for dimension in range(self.embedding_dimension):
                self.position_gradients[position][dimension] += (output_gradients[gradient_index][dimension])

    def update_parameters(self, learning_rate: float) -> None:
        if learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive.")
        if self.position_gradients is None:
            raise RuntimeError("backward() must be called before update_parameters().")

        self.optimizer.learning_rate = learning_rate
        self.optimizer.start_step()

        (self.position_matrix,self.position_first_moment,self.position_second_moment) = (
            self.optimizer.update(parameter=self.position_matrix,gradient=self.position_gradients,
            first_moment=self.position_first_moment,second_moment=self.position_second_moment))

        self.position_gradients = None