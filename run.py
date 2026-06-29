from src.tokenizer.bpe import BPETrainer


words = ["low", "lower", "lowest", "low", "lower"]

trainer = BPETrainer(words)

print("Corpus:")
print(trainer.corpus)

print("\nPair counts:")
print(trainer.count_pairs())
