from models.tiny_gpt import TinyGPT
from training.dataset import build_training_dataset
from training.train_tiny_gpt import train_tiny_gpt
from inference.decoder import Decoder

corpus = [
    "the cat sat on the mat",
    "the dog sat on the rug",
    "the cat slept on the rug",
    "the dog slept on the mat",
]

tokenizer, training_dataset = (
    build_training_dataset(
        corpus=corpus,
        num_merges=20,
        window_size=13,
        stride=6,
    )
)

print(training_dataset)

model = TinyGPT(
    vocabulary_size=len(tokenizer.vocab),
    embedding_dimension=8,
    maximum_sequence_length=32,
    number_of_heads=2,
    hidden_dimension=16,
    number_of_layers=2,
)

epoch_losses = train_tiny_gpt(model=model,training_dataset=training_dataset,epochs=500,learning_rate=0.005,batch_size=2)

prompt_ids = [tokenizer.vocab["<BOS>"]] + tokenizer.encode_ids("the")

decoder = Decoder(model)

# generated_ids = decoder.generate(
#     input_ids=prompt_ids,
#     maximum_new_tokens=20,
#     strategy="greedy",
#     eos_token_id=tokenizer.vocab["<EOS>"],
# )
# generated_ids = decoder.generate(
#     input_ids=prompt_ids,
#     maximum_new_tokens=20,
#     strategy="top_k",
#     top_k=2,
#     temperature=0.5,
#     repetition_penalty=None,
#     eos_token_id=tokenizer.vocab["<EOS>"],
# )
# generated_ids = decoder.generate(
#     input_ids=prompt_ids,
#     maximum_new_tokens=20,
#     strategy="top_p",
#     top_p=0.70,
#     temperature=0.5,
#     repetition_penalty=None,
#     eos_token_id=tokenizer.vocab["<EOS>"])
generated_ids = decoder.generate(
    input_ids=prompt_ids,
    maximum_new_tokens=20,
    strategy="beam",
    beam_width=3,
    eos_token_id=tokenizer.vocab["<EOS>"],
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

