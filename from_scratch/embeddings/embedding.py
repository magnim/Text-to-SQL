import random
from from_scratch.optimizers.adam import Adam


class Embedding:

    def __init__(self,vocab_size: int,embedding_dim: int):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.embedding_matrix = self._initialize_matrix()
        self.embedding_gradients: list[list[float]] | None = None
        self.last_token_ids: list[int] | None = None
        self.optimizer = Adam()

        self.embedding_first_moment = [[0.0 for _ in range(embedding_dim)] for _ in range(vocab_size)]

        self.embedding_second_moment = [[0.0 for _ in range(embedding_dim)] for _ in range(vocab_size)]

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

        if not isinstance(output_gradients, list):
            raise TypeError("output_gradients must be a list.")

        if not output_gradients:
            raise ValueError("output_gradients cannot be empty.")

        for row in output_gradients:
            if not isinstance(row, list):
                raise TypeError("Every output gradient row must be a list.")
            if len(row) != self.embedding_dim:
                raise ValueError(f"Each embedding gradient must contain {self.embedding_dim} values.")
            if any(not isinstance(value, (int, float)) for value in row):
                raise TypeError("Every embedding gradient value must be numeric.")

        self.embedding_gradients = [[0.0 for _ in range(self.embedding_dim)] for _ in range(self.vocab_size)]

        for token_id, gradient_row in zip(self.last_token_ids, output_gradients):
            for dimension in range(self.embedding_dim):
                self.embedding_gradients[token_id][dimension] += gradient_row[dimension]

    def update_parameters(self, learning_rate: float) -> None:
        if learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive.")

        if self.embedding_gradients is None:
            raise RuntimeError("backward() must be called before update_parameters().")

        self.optimizer.learning_rate = learning_rate
        self.optimizer.start_step()

        (self.embedding_matrix,self.embedding_first_moment,self.embedding_second_moment) = (
            self.optimizer.update(parameter=self.embedding_matrix,gradient=self.embedding_gradients,
                                  first_moment=self.embedding_first_moment,second_moment=self.embedding_second_moment))

        self.embedding_gradients = None

