from from_scratch.losses.cross_entropy import CrossEntropyLoss
from from_scratch.models.tiny_gpt import TinyGPT
from training.dataset import create_mini_batches
from from_scratch.optimizers.learning_rate_scheduler import LearningRateScheduler


def train_tiny_gpt(model: TinyGPT, training_dataset: list[dict], epochs: int, learning_rate: float, batch_size: int) -> list[float]:
    if not training_dataset:
        raise ValueError("training_dataset cannot be empty.")
    if epochs <= 0:
        raise ValueError("epochs must be positive.")
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive.")

    loss_function = CrossEntropyLoss()
    scheduler = LearningRateScheduler(initial_learning_rate=learning_rate, decay_factor=0.5, decay_every=100)
    epoch_losses = []

    for epoch in range(epochs):
        total_loss = 0.0
        processed_examples = 0

        mini_batches = create_mini_batches(training_dataset=training_dataset, batch_size=batch_size)
        current_learning_rate = scheduler.get_learning_rate(epoch)

        for batch in mini_batches:
            for example in batch:
                input_ids = example["input_ids"]
                target_ids = example["target_ids"]
                prompt_length = example["prompt_length"]
                schema_role_ids = example["schema_role_ids"]
                logits = model.forward(input_ids, schema_role_ids)

                # logits = model.forward(input_ids)

                sql_logits = logits[prompt_length - 1:]

                loss = loss_function.forward(sql_logits, target_ids)
                sql_gradients = loss_function.backward()

                logits_gradients = [[0.0 for _ in logits[0]] for _ in logits]

                for index, gradient in enumerate(sql_gradients):
                    logits_gradients[prompt_length - 1 + index] = gradient

                model.backward(logits_gradients)
                model.update_parameters(current_learning_rate)

                total_loss += loss
                processed_examples += 1

        average_loss = total_loss / processed_examples
        epoch_losses.append(average_loss)

        print(f"Epoch {epoch + 1}/{epochs} | LR: {current_learning_rate:.6f} | Average Loss: {average_loss:.6f}")

    return epoch_losses