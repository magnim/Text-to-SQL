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
        self.last_input = x
        if len(x) != self.input_size:
            raise ValueError(f"Expected {self.input_size} inputs, got {len(x)}.")
        output = 0.0
        for i in range(len(x)):
            output += self.weights[i] * x[i]
        return output+self.bias

    def backward(self,gradient: float) -> list[float]:
        self.weight_gradients = []
        for value in self.last_input:
            self.weight_gradients.append(value * gradient)
        self.bias_gradient = gradient
        input_gradients = []

        for weight in self.weights:
            input_gradients.append(weight * gradient)

        return input_gradients

    def update(
            self,
            learning_rate: float
    ) -> None:
        for i in range(len(self.weights)):
            self.weights[i] = (
                    self.weights[i]
                    - learning_rate * self.weight_gradients[i]
            )

        self.bias = (
                self.bias
                - learning_rate * self.bias_gradient
        )

