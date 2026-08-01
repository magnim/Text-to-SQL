from losses.cross_entropy import CrossEntropyLoss
from models.tiny_gpt import TinyGPT

def train_tiny_gpt(
    model: TinyGPT,
    training_dataset: list[dict[str, list[int]]],
    epochs: int,
    learning_rate: float
) -> list[float]:
    if not training_dataset:
        raise ValueError("training_dataset cannot be empty.")
    if epochs <= 0:
        raise ValueError("epochs must be positive.")
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive.")

    loss_function = CrossEntropyLoss()
    epoch_losses = []

    for epoch in range(epochs):
        total_loss = 0.0

        for example in training_dataset:
            input_ids = example["input_ids"]
            target_ids = example["target_ids"]

            logits = model.forward(input_ids)
            loss = loss_function.forward(logits, target_ids)
            logits_gradients = loss_function.backward()

            model.backward(logits_gradients)
            model.update_parameters(learning_rate)

            total_loss += loss

        average_loss = total_loss / len(training_dataset)
        epoch_losses.append(average_loss)

        print(
            f"Epoch {epoch + 1}/{epochs} "
            f"- Average Loss: {average_loss:.6f}"
        )

    return epoch_losses