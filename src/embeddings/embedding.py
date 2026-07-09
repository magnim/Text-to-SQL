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

    def forward(self,token_ids: list[int]) -> list[list[float]]:
        self.last_token_ids = token_ids
        embeddings  = []
        for token_id in token_ids:
            embeddings.append(self.embedding_matrix[token_id])
        return embeddings

    def backward(self, gradient: list[float]) -> None:
        token_id = self.last_token_ids[0]

        self.embedding_gradients[token_id] = gradient

    def update(self, learning_rate: float) -> None:
        for token_id in range(self.vocab_size):
            for i in range(self.embedding_dim):
                self.embedding_matrix[token_id][i] -= (learning_rate *
                                                       self.embedding_gradients[token_id][i])
                self.embedding_gradients[token_id][i] = 0.0
