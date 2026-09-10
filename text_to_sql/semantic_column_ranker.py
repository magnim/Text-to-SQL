import torch
import torch.nn as nn


class SemanticColumnRanker(nn.Module):
    def __init__(self, vocabulary_size: int, embedding_dimension: int = 64, hidden_dimension: int = 64):
        super().__init__()

        self.embedding = nn.Embedding(vocabulary_size, embedding_dimension)

        self.scorer = nn.Sequential(
            nn.Linear(embedding_dimension * 4, hidden_dimension),
            nn.ReLU(),
            nn.Linear(hidden_dimension, 1),
        )

    def _mean_pool(self, token_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        embeddings = self.embedding(token_ids)
        mask = mask.unsqueeze(-1).float()

        summed = (embeddings * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1.0)

        return summed / counts

    def forward(
        self,
        question_ids: torch.Tensor,
        question_mask: torch.Tensor,
        column_ids: torch.Tensor,
        column_mask: torch.Tensor,
    ) -> torch.Tensor:
        question_vector = self._mean_pool(question_ids, question_mask)
        column_vector = self._mean_pool(column_ids, column_mask)

        features = torch.cat([
            question_vector,
            column_vector,
            torch.abs(question_vector - column_vector),
            question_vector * column_vector,
        ], dim=-1)

        return self.scorer(features).squeeze(-1)



if __name__ == "__main__":
    model = SemanticColumnRanker(vocabulary_size=1200)

    question_ids = torch.tensor([
        [10, 20, 30, 40],
        [11, 21, 31, 0],
    ])

    question_mask = torch.tensor([
        [1, 1, 1, 1],
        [1, 1, 1, 0],
    ])

    column_ids = torch.tensor([
        [50, 0],
        [60, 61],
    ])

    column_mask = torch.tensor([
        [1, 0],
        [1, 1],
    ])

    scores = model(
        question_ids=question_ids,
        question_mask=question_mask,
        column_ids=column_ids,
        column_mask=column_mask,
    )

    print("Scores:", scores)
    print("Shape:", scores.shape)

    assert scores.shape == (2,)

    print("SemanticColumnRanker smoke test passed.")