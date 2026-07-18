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

    def backward(self, gradients: list[float]) -> list[float]:
        if len(gradients) != self.output_size:
            raise ValueError(
                f"Expected {self.output_size} gradients, got {len(gradients)}."
            )

        self.weight_gradients  = []
        self.bias_gradients = []
        input_gradients = [0.0] * self.input_size

        for weight_vector, gradient in zip(self.weights, gradients):
            weight_gradient = []

            for j in range(self.input_size):
                weight_gradient.append(self.last_input[j] * gradient)
                input_gradients[j] += weight_vector[j] * gradient

            self.weight_gradients.append(weight_gradient)
            self.bias_gradients.append(gradient)

        return input_gradients

    def update(
            self,
            learning_rate: float
    ) -> None:
        for neuron in range(self.output_size):
            for weight in range(self.input_size):
                self.weights[neuron][weight] -= (
                        learning_rate *
                        self.weight_gradients[neuron][weight]
                )
            self.biases[neuron] -= (
                    learning_rate *
                    self.bias_gradients[neuron]
            )

