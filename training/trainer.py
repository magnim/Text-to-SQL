class Trainer:

    def __init__(self,model,loss_fn,learning_rate: float):
        self.model = model
        self.loss_fn = loss_fn
        self.learning_rate = learning_rate

    def train(self,training_data: list[tuple[list[int], float]],epochs: int) -> None:
        for epoch in range(20):
            epoch_loss = 0.0
            for token_ids, target in training_data:
                prediction = self.model.forward(token_ids)
                loss = self.loss_fn.forward(prediction=prediction, target=target)
                gradient = self.loss_fn.backward(prediction=prediction, target=target)
                self.model.backward(gradient)
                self.model.update(learning_rate=self.learning_rate)
                epoch_loss += loss
            average_loss = epoch_loss / len(training_data)

            print(f"Epoch {epoch + 1:2d} | "f"Average Loss: {average_loss:.6f}")