import math


class Softmax:

    def forward(self,logits: list[float],) -> list[float]:
        if not logits:
            raise ValueError("logits cannot be empty.")
        maximum_logit = max(logits)
        exponentials = [math.exp(logit - maximum_logit) for logit in logits]
        total = sum(exponentials)
        probabilities = [value / total for value in exponentials]
        return probabilities