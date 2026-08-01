from models.tiny_gpt import TinyGPT
from src.tokenizer.bpe import BPETrainer
from training.train_tiny_gpt import train_tiny_gpt

corpus = [
    "the cat sat on the mat",
    "the dog sat on the rug",
    "the cat slept on the rug",
    "the dog slept on the mat",
]

training_words = []

for sentence in corpus:
    training_words.extend(sentence.split())

tokenizer = BPETrainer(training_words)
tokenizer.train(num_merges=20)

encoded_sequences = []

for sentence in corpus:
    sentence_ids = tokenizer.encode_ids(sentence)

    complete_ids = (
        [tokenizer.vocab["<BOS>"]]
        + sentence_ids
        + [tokenizer.vocab["<EOS>"]]
    )

    encoded_sequences.append(complete_ids)

training_dataset = []

for token_ids in encoded_sequences:
    training_dataset.append({
        "input_ids": token_ids[:-1],
        "target_ids": token_ids[1:],
    })

print(training_dataset)

model = TinyGPT(
    vocabulary_size=len(tokenizer.vocab),
    embedding_dimension=8,
    maximum_sequence_length=32,
    number_of_heads=2,
    hidden_dimension=16,
    number_of_layers=2,
)

epoch_losses = train_tiny_gpt(model=model,training_dataset=training_dataset,epochs=500,learning_rate=0.005)

prompt_ids = [tokenizer.vocab["<BOS>"]] + tokenizer.encode_ids("the")

generated_ids = model.generate(
    input_ids=prompt_ids,
    maximum_new_tokens=20,
    eos_token_id=tokenizer.vocab["<EOS>"]
)

id_to_token = {
    token_id: token
    for token, token_id in tokenizer.vocab.items()
}

print("Generated IDs:", generated_ids)
print(
    "Generated tokens:",
    [
        id_to_token.get(token_id, "<UNKNOWN>")
        for token_id in generated_ids
    ]
)
print("Generated text:", tokenizer.decode_ids(generated_ids))