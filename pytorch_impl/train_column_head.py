import torch
import torch.nn as nn
import random


def train_column_head(model, tokenizer, training_dataset: list[dict], epochs: int, learning_rate: float) -> list[float]:
    for parameter in model.parameters():
        parameter.requires_grad = False

    for parameter in model.column_query_projection.parameters():
        parameter.requires_grad = True

    for parameter in model.column_key_projection.parameters():
        parameter.requires_grad = True

    optimizer = torch.optim.Adam(
        [
            *model.column_query_projection.parameters(),
            *model.column_key_projection.parameters(),
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
            where_column = example["where_column"]

            if where_column is None:
                continue

            schema = example["schema"]

            candidate_columns = []

            for columns in schema.values():
                for column in columns:
                    if column not in candidate_columns:
                        candidate_columns.append(column)

            if where_column not in candidate_columns:
                raise ValueError(f"WHERE column '{where_column}' is not present in schema.")

            input_ids = torch.tensor(example["input_ids"][:example["prompt_length"]], dtype=torch.long)
            role_ids = torch.tensor(example["schema_role_ids"][:example["prompt_length"]], dtype=torch.long)

            column_scores = []

            for column in candidate_columns:
                column_token_ids = tokenizer.encode_ids(column)

                score = model.score_column(
                    input_ids=input_ids,
                    schema_role_ids=role_ids,
                    column_token_ids=column_token_ids,
                )

                column_scores.append(score)

            column_logits = torch.cat(column_scores).unsqueeze(0)

            correct_column_index = candidate_columns.index(where_column)
            column_target = torch.tensor([correct_column_index], dtype=torch.long)

            loss = loss_function(column_logits, column_target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            processed_examples += 1

        average_loss = total_loss / processed_examples
        epoch_losses.append(average_loss)

        print(
            f"Column Epoch {epoch + 1}/{epochs} | "
            f"Loss: {average_loss:.6f}"
        )

    return epoch_losses