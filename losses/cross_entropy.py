import math

class CrossEntropyLoss:
    def __init__(self):
        self.last_logits = []
        self.last_target = None

    def forward(self, logits: list[float], target: int) -> float:
        self.last_logits = logits
        self.last_target = target
        # Step 1: softmax
        exp_values = []
        for logit in logits:
            exp_values.append(math.exp(logit))
        total = sum(exp_values)
        probabilities = []
        for value in exp_values:
            probabilities.append(value / total)
        # Step 2: cross entropy
        correct_probability = probabilities[target]
        loss = -math.log(correct_probability)
        return loss

    def backward(self) -> list[float]:
        exp_values = []
        for logit in self.last_logits:
            exp_values.append(math.exp(logit))
        total = sum(exp_values)
        probabilities = []
        for value in exp_values:
            probabilities.append(value / total)
        gradients = probabilities.copy()
        gradients[self.last_target] -= 1.0
        return gradients