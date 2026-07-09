import random

from models.tiny_model import TinyModel
from losses.mse import MSE
from losses.cross_entropy import CrossEntropyLoss
from training.trainer import Trainer

random.seed(42)

model = TinyModel(
    vocab_size=3,
    embedding_dim=4,
    hidden_size=8
)

loss_fn = CrossEntropyLoss()

training_data = [
    ([0], 1),
    ([1], 2),
]

learning_rate = 0.01

trainer = Trainer(
    model=model,
    loss_fn=loss_fn,
    learning_rate=0.1
)

trainer.train(
    training_data=training_data,
    epochs=100
)