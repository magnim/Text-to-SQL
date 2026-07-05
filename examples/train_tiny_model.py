import random

from models.tiny_model import TinyModel
from losses.mse import MSE
from training.trainer import Trainer

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

trainer = Trainer(
    model=model,
    loss_fn=loss_fn,
    learning_rate=0.01
)

trainer.train(
    training_data=training_data,
    epochs=20
)