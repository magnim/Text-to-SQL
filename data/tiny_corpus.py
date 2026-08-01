CORPUS = [
    "the cat sat on the mat",
    "the dog sat on the rug",
    "the cat slept on the rug",
    "the dog slept on the mat"
]


def add_special_tokens(sentence):
    if not isinstance(sentence, str):
        raise TypeError("sentence must be a string.")

    sentence = sentence.strip()

    if not sentence:
        raise ValueError("sentence cannot be empty.")

    return f"<BOS> {sentence} <EOS>"


def prepare_corpus(corpus):
    if not isinstance(corpus, list):
        raise TypeError("corpus must be a list.")

    return [add_special_tokens(sentence) for sentence in corpus]