from src.tokenizer.bpe import BPETrainer

words = ["low", "lower", "lowest", "low", "lower"]

trainer = BPETrainer(words)

print("Initial Corpus:")
print(trainer.corpus)

trainer.train(3)

print("\nCorpus After 3 Merges:")
print(trainer.corpus)

print("\nMerge Rules:")
print(trainer.merge_rules)


print(trainer.encode("low"))
print(trainer.encode("lower"))
print(trainer.encode("lowest"))
print(trainer.encode("lowering"))
print(trainer.encode("yellow"))