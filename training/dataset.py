def create_training_example(token_ids):
    if not isinstance(token_ids, list):
        raise TypeError("token_ids must be a list.")

    if len(token_ids) < 2:
        raise ValueError("A sequence must contain at least two tokens.")

    input_ids = token_ids[:-1]
    target_ids = token_ids[1:]

    return input_ids, target_ids


def create_training_dataset(encoded_sequences):
    if not isinstance(encoded_sequences, list):
        raise TypeError("encoded_sequences must be a list.")

    if not encoded_sequences:
        raise ValueError("encoded_sequences cannot be empty.")

    training_dataset = []

    for token_ids in encoded_sequences:
        input_ids, target_ids = (create_training_example(token_ids))

        training_dataset.append({"input_ids": input_ids,"target_ids": target_ids})

    return training_dataset