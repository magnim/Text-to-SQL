import random

class Linear:

    def __init__(self, input_size):
        self.input_size = input_size
        self.weights = self._initialize_weights()
        self.bias = self._initialize_bias()

    def _initialize_weights(self) -> list[float]:
        weights = []
        for _ in range(self.input_size):
            value = random.uniform(-1, 1)
            weights.append(value)

        return weights

    def _initialize_bias(self) -> float:
        bias = random.uniform(-1, 1)

        return bias

    def forward(self,x: list[float]) -> float:
        if len(x) != self.input_size:
            raise ValueError(f"Expected {self.input_size} inputs, got {len(x)}.")
        output = 0.0
        for i in range(len(x)):
            output += self.weights[i] * x[i]
        return output+self.bias
