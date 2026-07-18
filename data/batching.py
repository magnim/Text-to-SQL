from typing import TypeVar
from dataclasses import dataclass
from data.padding import pad_sequences


@dataclass
class Batch:
    contexts: list[list[int]]
    targets: list[int]

T = TypeVar("T")

def create_batches(data: list[T],batch_size: int,drop_last: bool = False,) -> list[list[T]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero.")
    batches: list[list[T]] = []

    for start_index in range(0,len(data),batch_size,):
        batch = data[start_index:start_index + batch_size]
        if drop_last and len(batch) < batch_size:
            continue
        batches.append(batch)
    return batches


def collate_examples(examples: list[tuple[list[int], int]],pad_token_id: int = 0,target_length: int | None = None,) -> Batch:
    if not examples:
        return Batch(contexts=[],targets=[],)
    contexts = [context for context, _ in examples]
    targets = [target for _, target in examples]
    padded_contexts = pad_sequences(sequences=contexts,pad_token_id=pad_token_id,target_length=target_length,)
    return Batch(contexts=padded_contexts,targets=targets,)