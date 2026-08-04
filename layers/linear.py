import random
from optimizers.adam import Adam

class Linear:
    def __init__(self, input_dimension: int, output_dimension: int) -> None:
        if input_dimension <= 0:
            raise ValueError("input_dimension must be positive.")
        if output_dimension <= 0:
            raise ValueError("output_dimension must be positive.")

        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.weights = [
            [random.uniform(-0.1, 0.1) for _ in range(output_dimension)]
            for _ in range(input_dimension)
        ]
        self.biases = [0.0 for _ in range(output_dimension)]
        self.optimizer = Adam()

        self.weight_first_moment = [[0.0 for _ in range(output_dimension)] for _ in range(input_dimension)]

        self.weight_second_moment = [[0.0 for _ in range(output_dimension)] for _ in range(input_dimension)]

        self.bias_first_moment = [0.0 for _ in range(output_dimension)]

        self.bias_second_moment = [0.0 for _ in range(output_dimension)]
        self.weight_gradients: list[list[float]] | None = None
        self.bias_gradients: list[float] | None = None
        self.last_input: list[float] | list[list[float]] | None = None
        self.last_input_was_matrix = False

    def forward(self, inputs: list[float] | list[list[float]]) -> list[float] | list[list[float]]:
        self._validate_inputs(inputs)
        self.last_input_was_matrix = isinstance(inputs[0], list)

        if self.last_input_was_matrix:
            matrix_inputs = inputs
            self.last_input = [row.copy() for row in matrix_inputs]
            return [self._forward_vector(row) for row in matrix_inputs]

        vector_input = inputs
        self.last_input = vector_input.copy()
        return self._forward_vector(vector_input)

    def backward(self, output_gradients: list[float] | list[list[float]]) -> list[float] | list[list[float]]:
        if self.last_input is None:
            raise RuntimeError("forward() must be called before backward().")

        self._reset_gradients()

        if self.last_input_was_matrix:
            if not output_gradients or not isinstance(output_gradients[0], list):
                raise ValueError("Matrix input requires matrix output gradients.")

            matrix_inputs = self.last_input
            matrix_gradients = output_gradients

            if len(matrix_gradients) != len(matrix_inputs):
                raise ValueError("Gradient sequence length must match input sequence length.")

            return [
                self._backward_vector(input_row, gradient_row)
                for input_row, gradient_row in zip(matrix_inputs, matrix_gradients)
            ]

        if output_gradients and isinstance(output_gradients[0], list):
            raise ValueError("Vector input requires a vector output gradient.")

        vector_input = self.last_input
        vector_gradient = output_gradients
        return self._backward_vector(vector_input, vector_gradient)

    def update_parameters(self, learning_rate: float) -> None:
        if learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive.")

        if self.weight_gradients is None or self.bias_gradients is None:
            raise RuntimeError(
                "backward() must be called before update_parameters()."
            )

        self.optimizer.learning_rate = learning_rate
        self.optimizer.start_step()

        (
            self.weights,
            self.weight_first_moment,
            self.weight_second_moment,
        ) = self.optimizer.update(
            parameter=self.weights,
            gradient=self.weight_gradients,
            first_moment=self.weight_first_moment,
            second_moment=self.weight_second_moment,
        )

        (
            self.biases,
            self.bias_first_moment,
            self.bias_second_moment,
        ) = self.optimizer.update(
            parameter=self.biases,
            gradient=self.bias_gradients,
            first_moment=self.bias_first_moment,
            second_moment=self.bias_second_moment,
        )

    def _forward_vector(self, inputs: list[float]) -> list[float]:
        return [
            sum(inputs[input_index] * self.weights[input_index][output_index] for input_index in range(self.input_dimension) ) + self.biases[output_index]
            for output_index in range(self.output_dimension)
        ]

    def _backward_vector(self, inputs: list[float], output_gradient: list[float]) -> list[float]:
        if len(output_gradient) != self.output_dimension:
            raise ValueError("Output gradient must match output_dimension.")

        input_gradient = [0.0 for _ in range(self.input_dimension)]

        for input_index in range(self.input_dimension):
            for output_index in range(self.output_dimension):
                gradient = output_gradient[output_index]
                input_gradient[input_index] += gradient * self.weights[input_index][output_index]
                self.weight_gradients[input_index][output_index] += inputs[input_index] * gradient

        for output_index in range(self.output_dimension):
            self.bias_gradients[output_index] += output_gradient[output_index]

        return input_gradient

    def _reset_gradients(self) -> None:
        self.weight_gradients = [
            [0.0 for _ in range(self.output_dimension)]
            for _ in range(self.input_dimension)
        ]
        self.bias_gradients = [0.0 for _ in range(self.output_dimension)]

    def _validate_inputs(self, inputs: list[float] | list[list[float]]) -> None:
        if not isinstance(inputs, list):
            raise TypeError("inputs must be a list.")
        if not inputs:
            raise ValueError("inputs cannot be empty.")

        if isinstance(inputs[0], list):
            for row in inputs:
                self._validate_vector(row)
        else:
            self._validate_vector(inputs)

    def _validate_vector(self, values: list[float]) -> None:
        if len(values) != self.input_dimension:
            raise ValueError(f"Each input vector must contain {self.input_dimension} values.")
        if any(not isinstance(value, (int, float)) for value in values):
            raise TypeError("Every input value must be numeric.")