import random

from models.tiny_model import TinyModel
from losses.mse import MSE

random.seed(42)

model = TinyModel(
    vocab_size=5,
    embedding_dim=4
)

loss_fn = MSE()

training_data = [
    ([0], 1.0),
    ([1], 0.0),
    ([2], 0.5),
]

learning_rate = 0.01

for epoch in range(20):
    epoch_loss = 0.0
    for token_ids, target in training_data:
        prediction = model.forward(token_ids)
        loss = loss_fn.forward(prediction=prediction,target=target)
        gradient = loss_fn.backward(prediction=prediction,target=target)
        model.linear.backward(gradient)
        model.linear.update(learning_rate=learning_rate)
        epoch_loss += loss
    average_loss = epoch_loss / len(training_data)

    print(f"Epoch {epoch + 1:2d} | "f"Average Loss: {average_loss:.6f}")