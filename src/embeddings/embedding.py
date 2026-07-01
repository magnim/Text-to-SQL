import random


class Embedding:

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int
    ):

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.embedding_matrix = self._initialize_matrix()

    def _initialize_matrix(self) -> list[list[float]]:
        matrix = []

        for _ in range(self.vocab_size):
            vector = []

            for _ in range(self.embedding_dim):
                value = random.uniform(-1, 1)
                vector.append(value)

            matrix.append(vector)

        return matrix