import random
from utils.matrix import matrix_multiply, transpose


class FeedForwardNetwork:
    def __init__(self,embedding_dimension: int,hidden_dimension: int,) -> None:
        if embedding_dimension <= 0:
            raise ValueError("Embedding dimension must be positive.")

        if hidden_dimension <= 0:
            raise ValueError("Hidden dimension must be positive.")

        self.embedding_dimension = embedding_dimension
        self.hidden_dimension = hidden_dimension

        self.first_weights = self._initialize_matrix(embedding_dimension,hidden_dimension,)
        self.first_bias = [0.0] * hidden_dimension

        self.second_weights = self._initialize_matrix(hidden_dimension,embedding_dimension,)
        self.second_bias = [0.0] * embedding_dimension

        self.first_weights_gradient = None
        self.first_bias_gradient = None
        self.second_weights_gradient = None
        self.second_bias_gradient = None
        self.last_inputs = None
        self.last_first_projection = None
        self.last_activated = None
        self.last_output = None

    def _initialize_matrix(self,rows: int,columns: int) -> list[list[float]]:
        limit = (6.0 / (rows + columns)) ** 0.5
        return [[random.uniform(-limit, limit) for _ in range(columns)]for _ in range(rows)]

    def _validate_inputs(self, inputs: list[list[float]]) -> None:
        if not inputs:
            raise ValueError("Inputs cannot be empty.")
        for row in inputs:
            if len(row) != self.embedding_dimension:
                raise ValueError(f"Each input embedding must have {self.embedding_dimension} values.")

    def _add_bias(self,matrix: list[list[float]],bias: list[float]) -> list[list[float]]:
        return [[value + bias_value for value, bias_value in zip(row, bias)] for row in matrix]


    def _relu(self, matrix: list[list[float]]) -> list[list[float]]:
        return [[max(0.0, value) for value in row] for row in matrix]



    def forward(self,inputs: list[list[float]],) -> list[list[float]]:
        self._validate_inputs(inputs)
        self.last_inputs = inputs
        first_projection = matrix_multiply(inputs, self.first_weights)
        self.last_first_projection = self._add_bias(first_projection, self.first_bias)
        self.last_activated = self._relu(self.last_first_projection)
        second_projection = matrix_multiply(self.last_activated,self.second_weights,)
        self.last_output = self._add_bias(second_projection,self.second_bias,)
        return self.last_output

    def _relu_backward(self,output_gradient: list[list[float]],relu_inputs: list[list[float]]) -> list[list[float]]:
        return [[gradient if input_value > 0.0 else 0.0 for gradient, input_value in zip(gradient_row, input_row)] for gradient_row, input_row in zip(output_gradient, relu_inputs)]

    def _linear_backward(self,output_gradient: list[list[float]],layer_inputs: list[list[float]],weights: list[list[float]]) -> tuple[list[list[float]], list[list[float]], list[float]]:
        input_gradient = matrix_multiply(output_gradient, transpose(weights))
        weights_gradient = matrix_multiply(transpose(layer_inputs), output_gradient)
        bias_gradient = [sum(row[column] for row in output_gradient) for column in range(len(output_gradient[0]))]

        return input_gradient, weights_gradient, bias_gradient

    def _validate_output_gradient(self,output_gradient: list[list[float]]) -> None:
        if not output_gradient:
            raise ValueError("Output gradient cannot be empty.")

        if self.last_output is None:
            raise RuntimeError("Forward must be called before backward.")

        if len(output_gradient) != len(self.last_output):
            raise ValueError("Output gradient must match the output sequence length.")

        for row in output_gradient:
            if len(row) != self.embedding_dimension:
                raise ValueError(f"Each output-gradient row must have {self.embedding_dimension} values.")


    def backward(self, output_gradient: list[list[float]]) -> list[list[float]]:
        self._validate_output_gradient(output_gradient)

        activated_gradient, self.second_weights_gradient, self.second_bias_gradient = (self._linear_backward(output_gradient,self.last_activated,self.second_weights))

        first_projection_gradient = self._relu_backward(activated_gradient,self.last_first_projection)

        input_gradient, self.first_weights_gradient, self.first_bias_gradient = (self._linear_backward(first_projection_gradient,self.last_inputs,self.first_weights,))

        return input_gradient

    def update_parameters(self, learning_rate: float) -> None:
        if learning_rate <= 0.0:
            raise ValueError("Learning rate must be positive.")

        if self.first_weights_gradient is None:
            raise RuntimeError("Backward must be called before updating parameters.")

        for row in range(self.embedding_dimension):
            for column in range(self.hidden_dimension):
                self.first_weights[row][column] -= (learning_rate * self.first_weights_gradient[row][column])

        for index in range(self.hidden_dimension):
            self.first_bias[index] -= learning_rate * self.first_bias_gradient[index]

        for row in range(self.hidden_dimension):
            for column in range(self.embedding_dimension):
                self.second_weights[row][column] -= (learning_rate * self.second_weights_gradient[row][column])

        for index in range(self.embedding_dimension):
            self.second_bias[index] -= learning_rate * self.second_bias_gradient[index]