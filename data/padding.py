def pad_sequence(token_ids: list[int], target_length: int, pad_token_id: int = 0, ) -> list[int]:
    if target_length < 0:
        raise ValueError("target_length cannot be negative.")

    if len(token_ids) > target_length:
        raise ValueError("Sequence length cannot exceed target_length.")

    padding_needed = target_length - len(token_ids)

    return token_ids + [pad_token_id] * padding_needed


def pad_sequences(sequences: list[list[int]], pad_token_id: int = 0, target_length: int | None = None, ) -> list[list[int]]:

    if not sequences:
        return []

    if target_length is None:
        target_length = max(len(sequence) for sequence in sequences)

    return [pad_sequence(token_ids=sequence, target_length=target_length, pad_token_id=pad_token_id, ) for sequence in
            sequences]
