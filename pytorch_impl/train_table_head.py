import torch
import torch.nn as nn
import random


def train_table_head(model, tokenizer, training_dataset: list[dict], epochs: int, learning_rate: float) -> list[float]:
    for parameter in model.parameters():
        parameter.requires_grad = False

    for parameter in model.table_query_projection.parameters():
        parameter.requires_grad = True

    for parameter in model.table_key_projection.parameters():
        parameter.requires_grad = True

    optimizer = torch.optim.Adam(
        [
            *model.table_query_projection.parameters(),
            *model.table_key_projection.parameters(),
        ],
        lr=learning_rate,
    )

    loss_function = nn.CrossEntropyLoss()
    epoch_losses = []

    for epoch in range(epochs):
        total_loss = 0.0
        processed_examples = 0

        shuffled_dataset = training_dataset.copy()
        random.shuffle(shuffled_dataset)

        for example in shuffled_dataset:
            schema = example["schema"]
            correct_table = example["table_name"]

            # One table means there is no table-selection decision to learn.
            if len(schema) <= 1:
                continue

            candidate_tables = list(schema.keys())

            if correct_table not in candidate_tables:
                raise ValueError(
                    f"Table '{correct_table}' is not present in schema."
                )

            prompt_length = example["prompt_length"]

            input_ids = torch.tensor(
                example["input_ids"][:prompt_length],
                dtype=torch.long,
            )

            role_ids = torch.tensor(
                example["schema_role_ids"][:prompt_length],
                dtype=torch.long,
            )

            table_scores = []

            for table in candidate_tables:
                table_token_ids = tokenizer.encode_ids(table)

                score = model.score_table(
                    input_ids=input_ids,
                    schema_role_ids=role_ids,
                    table_token_ids=table_token_ids,
                )

                table_scores.append(score)

            table_logits = torch.cat(table_scores).unsqueeze(0)

            correct_table_index = candidate_tables.index(correct_table)
            table_target = torch.tensor(
                [correct_table_index],
                dtype=torch.long,
            )

            loss = loss_function(table_logits, table_target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            processed_examples += 1

        if processed_examples == 0:
            raise ValueError(
                "No multi-table training examples were found."
            )

        average_loss = total_loss / processed_examples
        epoch_losses.append(average_loss)

        print(
            f"Table Epoch {epoch + 1}/{epochs} | "
            f"Loss: {average_loss:.6f} | "
            f"Examples: {processed_examples}"
        )

    return epoch_losses