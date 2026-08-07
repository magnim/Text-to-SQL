class RepetitionPenalty:
    def __init__(self, penalty: float) -> None:
        if not isinstance(penalty, (int, float)):
            raise TypeError("penalty must be numeric.")
        if penalty <= 1.0:
            raise ValueError("penalty must be greater than 1.")
        self.penalty = penalty

    def forward(self,logits: list[float],generated_ids: list[int]) -> list[float]:
        if not isinstance(logits, list):
            raise TypeError("logits must be a list.")
        if not logits:
            raise ValueError("logits cannot be empty.")
        if any(not isinstance(logit, (int, float)) for logit in logits):
            raise TypeError("Every logit must be numeric.")
        if not isinstance(generated_ids, list):
            raise TypeError("generated_ids must be a list.")
        adjusted_logits = logits.copy()
        unique_token_ids = set(generated_ids)
        for token_id in unique_token_ids:
            if not isinstance(token_id, int):
                raise TypeError("Every generated token ID must be an integer.")
            if token_id < 0 or token_id >= len(adjusted_logits):
                raise ValueError(f"Token ID {token_id} is outside the logits range.")
            if adjusted_logits[token_id] > 0.0:
                adjusted_logits[token_id] /= self.penalty
            else:
                adjusted_logits[token_id] *= self.penalty
        return adjusted_logits

