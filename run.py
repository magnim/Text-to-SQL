from src.tokenizer.bpe import BPETrainer
from layers.linear import Linear
from models.tiny_model import TinyModel

words = ["low", "lower", "lowest", "low", "lower"]

trainer = BPETrainer(words)

print("Initial Corpus:")
print(trainer.corpus)

trainer.train(3)

print("\nCorpus After 3 Merges:")
print(trainer.corpus)

print("\nMerge Rules:")
print(trainer.merge_rules)


print(trainer.vocab)

print("Tokens:")
print(trainer.encode("lowest"))

print("\nToken IDs:")
print(trainer.encode_ids("lowest"))

from src.embeddings.embedding import Embedding


embedding = Embedding(
    vocab_size=5,
    embedding_dim=3
)

print(embedding.embedding_matrix)
print('#'*20)
vectors = embedding.forward([0, 2, 4])

print(vectors)


linear = Linear(input_size=2)
linear.weights = [3, 4]
linear.bias = 10
prediction = linear.forward([2, 5])

print(prediction)



linear.weights = [3, 4]
linear.bias = 10

print(linear.forward([0, 0]))


linear.weights = [0.5, 0.2]
linear.bias = 0.1

print(linear.forward([2.0, 4.0]))


model = TinyModel(
    vocab_size=3,
    embedding_dim=2
)

prediction = model.forward([0])

print(prediction)