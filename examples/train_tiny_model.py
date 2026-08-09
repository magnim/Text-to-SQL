import random

from data.text_dataset import TextDataset
from data.data_loader import DataLoader
from from_scratch.losses.cross_entropy import CrossEntropyLoss
from from_scratch.models.tiny_model import TinyModel
from from_scratch.training.trainer import Trainer

random.seed(42)

model = TinyModel(vocab_size=3,embedding_dim=4,hidden_size=8)

loss_fn = CrossEntropyLoss()

token_ids = [0,1,2,]

dataset = TextDataset(token_ids=token_ids,context_size=1)

data_loader = DataLoader(dataset,batch_size=2,shuffle=True)

trainer = Trainer(model=model,loss_fn=loss_fn,learning_rate=0.1)

trainer.train(data_loader,100)