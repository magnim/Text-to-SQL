import random
import torch
import torch.nn as nn


def pretrain_tiny_gpt(model,training_dataset: list[dict],epochs: int,learning_rate: float,batch_size: int) -> list[float]:

    if not training_dataset:
        raise ValueError("training_dataset cannot be empty.")

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    loss_function = nn.CrossEntropyLoss()

    epoch_losses = []

    for epoch in range(epochs):
        shuffled_dataset = training_dataset.copy()
        random.shuffle(shuffled_dataset)
        total_loss = 0.0
        batch_count = 0
        for start in range(0, len(shuffled_dataset), batch_size):
            batch = shuffled_dataset[start:start + batch_size]

            input_ids = torch.tensor([example["input_ids"] for example in batch],dtype=torch.long)
            target_ids = torch.tensor([example["target_ids"] for example in batch],dtype=torch.long)
            optimizer.zero_grad()
            logits = model(input_ids)
            loss = loss_function(logits.reshape(-1, logits.size(-1)),target_ids.reshape(-1))

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batch_count += 1
        average_loss = total_loss / len(shuffled_dataset)
        epoch_losses.append(average_loss)

        lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"LR: {lr:.6f} | "
            f"Average Loss: {average_loss:.6f}"
        )

        scheduler.step()

    return epoch_losses