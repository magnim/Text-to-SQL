from src.tokenizer.bpe import BPETrainer

words = ["low", "lower", "lowest", "low", "lower"]

trainer = BPETrainer(words)

print("Corpus:")
print(trainer.corpus)

print("\nPair Counts:")
print(trainer.count_pairs())

print("\nBest Pair:")
print(trainer.find_best_pair())
