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

        self.linear = Linear(
            input_size=embedding_dim
        )

        self.relu = ReLU()

    def forward(
            self,
            token_ids: list[int]
    ) -> float:
        embeddings = self.embedding.forward(token_ids)

        linear_output = self.linear.forward(embeddings[0])
        activated_output = self.relu.forward(linear_output)

        return activated_output

    def backward(self,gradient: float) -> None:
        gradient = self.relu.backward(gradient)

        self.linear.backward(gradient)


    def update(self,learning_rate: float):
        self.linear.update(learning_rate)
