from src.tokenizer.bpe import BPETrainer

words = ["low", "lower", "lowest", "low", "lower"]

trainer = BPETrainer(words)

print("Corpus:")
print(trainer.corpus)

print("\nPair Counts:")
print(trainer.count_pairs())

best_pair = trainer.find_best_pair()
print("\nBest pair:")
print(best_pair)

trainer.merge_pair(best_pair)

print("\nAfter merge:")
print(trainer.corpus)
