from src.embeddings.embedding import Embedding
from layers.linear import Linear

class TinyModel:

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int
    ):
        self.embedding = Embedding(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim
        )

        self.linear = Linear(
            input_size=embedding_dim
        )

    def forward(
            self,
            token_ids: list[int]
    ) -> float:
        embeddings = self.embedding.forward(token_ids)

        prediction = self.linear.forward(
            embeddings[0]
        )

        return prediction

    def backward(self,gradient: float) -> None:
        self.linear.backward(gradient)


    def update(self,learning_rate: float):
        self.linear.update(learning_rate)
