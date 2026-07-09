from src.embeddings.embedding import Embedding
from layers.linear import Linear
from layers.relu import ReLU

class TinyModel:

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_size: int
    ):
        self.embedding = Embedding(vocab_size=vocab_size,embedding_dim=embedding_dim)
        self.linear1 = Linear(input_size=embedding_dim,output_size=hidden_size)
        self.relu = ReLU()
        self.linear2 = Linear(input_size=hidden_size,output_size=vocab_size)

    def forward(self, token_ids: list[int]) -> list[float]:
        embeddings = self.embedding.forward(token_ids)
        hidden_vectors = []
        for embedding in embeddings:
            hidden = self.linear1.forward(embedding)
            hidden = self.relu.forward(hidden)
            hidden_vectors.append(hidden)
        last_hidden = hidden_vectors[-1]
        logits = self.linear2.forward(last_hidden)
        return logits

    def backward(self, gradients: list[float]) -> None:
        gradients = self.linear2.backward(gradients)
        gradients = self.relu.backward(gradients)
        gradients = self.linear1.backward(gradients)
        self.embedding.backward(gradients)

    def update(self, learning_rate: float) -> None:
        self.embedding.update(learning_rate)
        self.linear1.update(learning_rate)
        self.linear2.update(learning_rate)

    def predict_next(self,token_ids: list[int]) -> int:
        logits = self.forward(token_ids)
        best_index = 0
        for i in range(1, len(logits)):
            if logits[i] > logits[best_index]:
                best_index = i
        return best_index

