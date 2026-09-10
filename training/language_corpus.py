from datasets import load_dataset


def load_language_corpus(limit: int = 5000) -> list[str]:
    dataset = load_dataset(
        "Salesforce/wikitext",
        "wikitext-2-raw-v1",
        split="train"
    )

    corpus = [text.strip() for text in dataset["text"] if text.strip()]
    return corpus[:limit]