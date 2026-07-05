import random

from models.tiny_model import TinyModel
from losses.mse import MSE

random.seed(42)

model = TinyModel(
    vocab_size=5,
    embedding_dim=4
)

loss_fn = MSE()

token_ids = [0]

target = 1.0

learning_rate = 0.01

for epoch in range(150):
    prediction = model.forward(token_ids)

    loss = loss_fn.forward(
        prediction=prediction,
        target=target
    )

    gradient = loss_fn.backward(
        prediction=prediction,
        target=target
    )

    model.linear.backward(gradient)

    model.linear.update(
        learning_rate=learning_rate
    )

    print(
        f"Epoch {epoch + 1} | "
        f"Prediction: {prediction:.4f} | "
        f"Loss: {loss:.4f}"
    )