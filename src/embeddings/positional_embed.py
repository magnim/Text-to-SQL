import random


class PositionalEmbedding:

    def __init__(self,max_sequence_length,embedding_dimension):

        if max_sequence_length <= 0:
            raise ValueError("Maximum sequence length must be positive.")

        if embedding_dimension <= 0:
            raise ValueError("Embedding dimension must be positive.")

        self.max_sequence_length = max_sequence_length
        self.embedding_dimension = embedding_dimension
        self.position_matrix = [[random.uniform(-1, 1) for _ in range(embedding_dimension)]for _ in range(max_sequence_length)]
        self.last_positions = None
        self.position_gradients = None

    def forward(self, sequence_length):
        if sequence_length <= 0:
            raise ValueError("Sequence length must be positive.")
        if sequence_length > self.max_sequence_length:
            raise ValueError("Sequence length exceeds maximum.")
        self.last_positions = list(range(sequence_length))
        return [self.position_matrix[position].copy() for position in self.last_positions]

    def _validate_backward(self):

        if self.last_positions is None:
            raise RuntimeError("forward() must be called before backward().")

    def backward(self, output_gradients: list[list[float]]) -> None:
        if self.last_positions is None:
            raise RuntimeError("forward() must be called before backward().")

        if len(output_gradients) != len(self.last_positions):
            raise ValueError("Gradient sequence length must match the used positions.")

        for row in output_gradients:
            if len(row) != self.embedding_dimension:
                raise ValueError(f"Each positional gradient must contain {self.embedding_dimension} values.")

        self.position_gradients = [[0.0 for _ in range(self.embedding_dimension)] for _ in range(self.max_sequence_length)]

        for gradient_index, position in enumerate(self.last_positions):
            for dimension in range(self.embedding_dimension):
                self.position_gradients[position][dimension] += (output_gradients[gradient_index][dimension])

    def update_parameters(self, learning_rate: float) -> None:
        if learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive.")
        if self.position_gradients is None:
            raise RuntimeError("backward() must be called before update_parameters().")

        for position in self.last_positions:
            for dimension in range(self.embedding_dimension):
                self.position_matrix[position][dimension] -= (
                        learning_rate * self.position_gradients[position][dimension]
                )