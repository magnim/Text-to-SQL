import random


class Embedding:

    def __init__(self,vocab_size: int,embedding_dim: int):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.embedding_matrix = self._initialize_matrix()
        self.embedding_gradients = [
            [0.0 for _ in range(embedding_dim)]
            for _ in range(vocab_size)]

    def _initialize_matrix(self) -> list[list[float]]:
        matrix = []
        for _ in range(self.vocab_size):
            vector = []
            for _ in range(self.embedding_dim):
                value = random.uniform(-1, 1)
                vector.append(value)
            matrix.append(vector)
        return matrix

    def _validate_token_ids(self, token_ids: list[int]) -> None:
        if not isinstance(token_ids, list):
            raise TypeError("token_ids must be a list.")

        if not token_ids:
            raise ValueError("token_ids cannot be empty.")

        for token_id in token_ids:
            if not isinstance(token_id, int):
                raise TypeError("Every token ID must be an integer.")

            if token_id < 0 or token_id >= self.vocab_size:
                raise ValueError(f"Token ID {token_id} is outside the vocabulary range.")

    def forward(self, token_ids: list[int]) -> list[list[float]]:
        self._validate_token_ids(token_ids)
        self.last_token_ids = token_ids.copy()
        return [self.embedding_matrix[token_id].copy() for token_id in token_ids]

    def backward(self, output_gradients: list[list[float]]) -> None:
        if self.last_token_ids is None:
            raise RuntimeError("forward() must be called before backward().")

        if len(output_gradients) != len(self.last_token_ids):
            raise ValueError("Gradient sequence length must match the number of token IDs.")

        for row in output_gradients:
            if len(row) != self.embedding_dim:
                raise ValueError(
                    f"Each embedding gradient must contain {self.embedding_dim} values."
                )

        self.embedding_gradients = [
            [0.0 for _ in range(self.embedding_dim)]
            for _ in range(self.vocab_size)
        ]

        for token_id, gradient_row in zip(self.last_token_ids, output_gradients):
            for dimension in range(self.embedding_dim):
                self.embedding_gradients[token_id][dimension] += gradient_row[dimension]

    def update_parameters(self, learning_rate: float) -> None:
        if learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive.")
        if self.embedding_gradients is None:
            raise RuntimeError("backward() must be called before update_parameters().")

        for token_id in range(self.vocab_size):
            for dimension in range(self.embedding_dim):
                self.embedding_matrix[token_id][dimension] -= (
                        learning_rate * self.embedding_gradients[token_id][dimension]
                )

