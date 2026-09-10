import json
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader

from src.tokenizer.bpe import BPETrainer
from text_to_sql.semantic_column_ranker import SemanticColumnRanker


PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRAIN_PATH = PROJECT_ROOT / "data" / "column_linking" / "train.jsonl"
DEV_PATH = PROJECT_ROOT / "data" / "column_linking" / "dev.jsonl"
TOKENIZER_PATH = PROJECT_ROOT / "tokenizer_balanced.json"
MODEL_OUTPUT_PATH = PROJECT_ROOT / "semantic_column_ranker.pt"

QUESTION_MAX_LENGTH = 64
COLUMN_MAX_LENGTH = 16
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 0.001


class ColumnLinkingDataset(Dataset):
    def __init__(self, path: Path, tokenizer):
        self.rows = []

        with open(path, "r", encoding="utf-8") as file:
            for line in file:
                self.rows.append(json.loads(line))

        self.tokenizer = tokenizer
        self.pad_id = tokenizer.vocab["<PAD>"]

    def __len__(self):
        return len(self.rows)

    def _encode(self, text: str, max_length: int):
        token_ids = self.tokenizer.encode_ids(text)[:max_length]
        mask = [1] * len(token_ids)

        padding = max_length - len(token_ids)

        token_ids += [self.pad_id] * padding
        mask += [0] * padding

        return token_ids, mask

    def __getitem__(self, index):
        row = self.rows[index]

        question_ids, question_mask = self._encode(
            row["context"],
            QUESTION_MAX_LENGTH,
        )

        column_text = f'{row["table"]} {row["column"]}'

        column_ids, column_mask = self._encode(
            column_text,
            COLUMN_MAX_LENGTH,
        )

        return {
            "question_ids": torch.tensor(question_ids, dtype=torch.long),
            "question_mask": torch.tensor(question_mask, dtype=torch.long),
            "column_ids": torch.tensor(column_ids, dtype=torch.long),
            "column_mask": torch.tensor(column_mask, dtype=torch.long),
            "label": torch.tensor(row["label"], dtype=torch.float),
        }


def evaluate(model, data_loader, loss_function):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in data_loader:
            scores = model(
                question_ids=batch["question_ids"],
                question_mask=batch["question_mask"],
                column_ids=batch["column_ids"],
                column_mask=batch["column_mask"],
            )

            labels = batch["label"]

            loss = loss_function(scores, labels)
            total_loss += loss.item()

            predictions = (torch.sigmoid(scores) >= 0.5).float()

            correct += (predictions == labels).sum().item()
            total += labels.numel()

    return total_loss / len(data_loader), correct / total


def main():
    tokenizer = BPETrainer.load(TOKENIZER_PATH)

    train_dataset = ColumnLinkingDataset(TRAIN_PATH, tokenizer)
    dev_dataset = ColumnLinkingDataset(DEV_PATH, tokenizer)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    dev_loader = DataLoader(
        dev_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    model = SemanticColumnRanker(
        vocabulary_size=len(tokenizer.vocab),
        embedding_dimension=64,
        hidden_dimension=64,
    )

    positive_count = sum(row["label"] == 1 for row in train_dataset.rows)
    negative_count = sum(row["label"] == 0 for row in train_dataset.rows)

    positive_weight = torch.tensor(
        negative_count / positive_count,
        dtype=torch.float,
    )

    loss_function = torch.nn.BCEWithLogitsLoss(
        pos_weight=positive_weight,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    print("Train rows:", len(train_dataset))
    print("Dev rows:", len(dev_dataset))
    print("Positive weight:", positive_weight.item())


    best_dev_loss = float("inf")
    for epoch in range(EPOCHS):
        model.train()
        total_train_loss = 0.0

        for batch in train_loader:
            optimizer.zero_grad()

            scores = model(
                question_ids=batch["question_ids"],
                question_mask=batch["question_mask"],
                column_ids=batch["column_ids"],
                column_mask=batch["column_mask"],
            )

            labels = batch["label"]

            loss = loss_function(scores, labels)

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        train_loss = total_train_loss / len(train_loader)

        dev_loss, dev_accuracy = evaluate(model,dev_loader,loss_function)
        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            torch.save(model.state_dict(), MODEL_OUTPUT_PATH)

        print(
            f"Epoch {epoch + 1:02d} | "
            f"train_loss={train_loss:.4f} | "
            f"dev_loss={dev_loss:.4f} | "
            f"dev_accuracy={dev_accuracy:.4f}"
        )

    # torch.save(model.state_dict(), MODEL_OUTPUT_PATH)

    print("\nSaved model:", MODEL_OUTPUT_PATH)


if __name__ == "__main__":
    main()