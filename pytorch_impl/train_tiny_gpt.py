import torch
import torch.nn as nn
import random
from datetime import datetime as dt


def train_tiny_gpt(model, tokenizer, training_dataset: list[dict], epochs: int, learning_rate: float, batch_size: int) -> list[float]:
    if not training_dataset:
        raise ValueError("training_dataset cannot be empty.")
    if epochs <= 0:
        raise ValueError("epochs must be positive.")
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive.")

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=20,gamma=0.5)

    sql_loss_function = nn.CrossEntropyLoss()
    operator_loss_function = nn.CrossEntropyLoss()
    column_loss_function = nn.CrossEntropyLoss()
    epoch_losses = []

    for epoch in range(epochs):
        total_loss = 0.0
        total_sql_loss = 0.0
        total_operator_loss = 0.0
        # total_column_loss = 0.0
        processed_examples = 0
        shuffled_dataset = training_dataset.copy()
        random.shuffle(shuffled_dataset)
        for start in range(0, len(shuffled_dataset), batch_size):
            batch = shuffled_dataset[start:start + batch_size]
            for example in batch:
                input_ids = torch.tensor(example["input_ids"], dtype=torch.long)
                target_ids = torch.tensor(example["target_ids"], dtype=torch.long)
                schema_role_ids = torch.tensor(example["schema_role_ids"], dtype=torch.long)
                prompt_length = example["prompt_length"]
                optimizer.zero_grad()
                logits = model(input_ids, schema_role_ids)
                # operator_logits = model.predict_operator(input_ids, schema_role_ids)
                operator_logits = model.predict_operator(input_ids[:prompt_length],schema_role_ids[:prompt_length])
                operator_target = torch.tensor([example["operator_label"]], dtype=torch.long)
                sql_logits = logits[0, prompt_length - 1:]
                sql_loss = sql_loss_function(sql_logits, target_ids)
                operator_loss = operator_loss_function(operator_logits,operator_target)

                # column_loss = sql_loss.new_tensor(0.0)
                # where_column = example["where_column"]
                # if where_column is not None:
                #     schema = example["schema"]
                #     candidate_columns = []
                #     for columns in schema.values():
                #         for column in columns:
                #             if column not in candidate_columns:
                #                 candidate_columns.append(column)
                #     if where_column not in candidate_columns:
                #         raise ValueError(f"WHERE column '{where_column}' is not present in schema.")
                #     prompt_input_ids = input_ids[:prompt_length]
                #     prompt_role_ids = schema_role_ids[:prompt_length]
                #     column_scores = []
                #     for column in candidate_columns:
                #         column_token_ids = tokenizer.encode_ids(column)
                #         score = model.score_column(input_ids=prompt_input_ids,schema_role_ids=prompt_role_ids,column_token_ids=column_token_ids)
                #         column_scores.append(score)
                #     column_logits = torch.cat(column_scores).unsqueeze(0)
                #     correct_column_index = candidate_columns.index(where_column)
                #     column_target = torch.tensor([correct_column_index],dtype=torch.long)
                #     column_loss = column_loss_function(column_logits,column_target)

                loss = sql_loss + operator_loss #+ column_loss
                total_sql_loss += sql_loss.item()
                total_operator_loss += operator_loss.item()
                # total_column_loss += column_loss.item()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                processed_examples += 1
        average_loss = total_loss / processed_examples
        average_sql_loss = total_sql_loss / processed_examples
        average_operator_loss = total_operator_loss / processed_examples
        # average_column_loss = total_column_loss / processed_examples
        epoch_losses.append(average_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"LR: {current_lr:.6f} | "
            f"Loss: {average_loss:.6f} | "
            f"SQL: {average_sql_loss:.6f} | "
            f"Operator: {average_operator_loss:.6f} | "
            # f"Column: {average_column_loss:.6f}| "
            f"Completed at >> {dt.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        # print(f"Epoch {epoch} completed at {dt.now():%Y-%m-%d %H:%M:%S}")
        # scheduler.step()
    return epoch_losses