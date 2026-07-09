class Trainer:

    def __init__(self, model, loss_fn, learning_rate: float):
        self.model = model
        self.loss_fn = loss_fn
        self.learning_rate = learning_rate

    def train(self,training_data: list[tuple[list[int], int]],epochs: int) -> None:
        for epoch in range(epochs):
            epoch_loss = 0.0
            for token_ids, target in training_data:
                logits = self.model.forward(token_ids)
                loss = self.loss_fn.forward(logits, target)
                gradients = self.loss_fn.backward()
                self.model.backward(gradients)
                self.model.update(self.learning_rate)
                epoch_loss += loss
            average_loss = epoch_loss / len(training_data)
            print(f"Epoch {epoch + 1:2d} | "f"Average Loss: {average_loss:.6f}")