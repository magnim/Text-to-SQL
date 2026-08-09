from data.data_loader import DataLoader


class Trainer:

    def __init__(self, model, loss_fn, learning_rate: float):
        self.model = model
        self.loss_fn = loss_fn
        self.learning_rate = learning_rate

    def train(self, data_loader: DataLoader, epochs: int) -> None:
        for epoch in range(epochs):
            epoch_loss = 0.0
            total_examples = 0

            for batch in data_loader:
                for context, target in zip(batch.contexts, batch.targets):
                    logits = self.model.forward(context)
                    loss = self.loss_fn.forward(logits, target)
                    gradients = self.loss_fn.backward()

                    self.model.backward(gradients)
                    self.model.update(self.learning_rate)

                    epoch_loss += loss
                    total_examples += 1

            average_loss = epoch_loss / total_examples
            print(f"Epoch {epoch + 1}: Loss = {average_loss:.6f}")