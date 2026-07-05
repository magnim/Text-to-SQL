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

    def backward(self,gradient: float) -> None:
        self.weight_gradients = []
        for i in range(len(self.last_input)):
            self.weight_gradients.append(self.last_input[i]*gradient)

        self.bias_gradient = gradient

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

