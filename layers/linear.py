import random

class Linear:

    def __init__(self, input_size: int, output_size: int):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = self._initialize_weights()
        self.biases = self._initialize_bias()

    def _initialize_weights(self) -> list[list[float]]:
        weights = []
        for _ in range(self.output_size):
            weight = []
            for _ in range(self.input_size):
                value = random.uniform(-1, 1)
                weight.append(value)
            weights.append(weight)

        return weights

    def _initialize_bias(self) -> list[float]:
        biases = []
        for _ in range(self.output_size):
            bias = random.uniform(-1, 1)
            biases.append(bias)

        return biases

    def forward(self,x: list[float]) -> list[float]:
        if len(x) != self.input_size:
            raise ValueError(f"Expected {self.input_size} inputs, got {len(x)}.")
        self.last_input = x
        outputs = []
        for weight_vector, bias in zip(self.weights, self.biases):
            output = 0.0

            for j in range(self.input_size):
                output += weight_vector[j] * x[j]
            outputs.append(output+bias)
        return outputs

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

