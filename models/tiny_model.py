from src.embeddings.embedding import Embedding
from layers.linear import Linear
from layers.relu import ReLU

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

        self.linear1 = Linear(
            input_size=embedding_dim
        )

        self.relu = ReLU()
        self.linear2 = Linear(input_size=1)

    def forward(
            self,
            token_ids: list[int]
    ) -> float:
        embeddings = self.embedding.forward(token_ids)

        linear_output = self.linear1.forward(embeddings[0])
        activated_output = self.relu.forward(linear_output)
        prediction = self.linear2.forward([activated_output])

        return prediction

    def backward(self, gradient: float) -> None:
        gradient = self.linear2.backward(gradient)
        gradient = self.relu.backward(gradient[0])
        self.linear1.backward(gradient)

    def update(self, learning_rate: float) -> None:
        self.linear1.update(learning_rate)
        self.linear2.update(learning_rate)
