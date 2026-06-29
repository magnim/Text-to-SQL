from src.tokenizer.bpe import BPETrainer

words = ["low", "lower", "lowest", "low", "lower"]

trainer = BPETrainer(words)

print("Initial Corpus:")
print(trainer.corpus)

trainer.train(3)

print("\nCorpus After 3 Merges:")
print(trainer.corpus)