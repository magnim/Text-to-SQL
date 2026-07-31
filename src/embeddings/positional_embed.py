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

    def backward(self, output_gradient):

        self._validate_backward()

        if len(output_gradient) != len(self.last_positions):
            raise ValueError("Gradient length does not match sequence length.")

        self.position_gradients = [gradient.copy() for gradient in output_gradient]

        return [gradient.copy() for gradient in output_gradient]

    def update_parameters(self, learning_rate):

        if self.position_gradients is None:
            raise RuntimeError("backward() must be called before update.")

        for row_index, position in enumerate(self.last_positions):
            for column in range(self.embedding_dimension):
                self.position_matrix[position][column] -= (learning_rate * self.position_gradients[row_index][column])