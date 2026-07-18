def create_context_windows(token_ids: list[int],context_size: int) -> list[tuple[list[int], int]]:
    if context_size <= 0:
        raise ValueError(
            "context_size must be greater than zero."
        )

    if len(token_ids) <= context_size:
        return []

    training_examples: list[tuple[list[int], int]] = []

    for start_index in range(len(token_ids) - context_size):
        end_index = start_index + context_size

        context = token_ids[
            start_index:end_index
        ]

        target = token_ids[end_index]

        training_examples.append(
            (context, target)
        )

    return training_examples

token_ids = [10,20,30,40,50,60,]

examples = create_context_windows(
    token_ids=token_ids,
    context_size=3,
)

for context, target in examples:
    print(
        f"Input: {context} | "
        f"Target: {target}"
    )