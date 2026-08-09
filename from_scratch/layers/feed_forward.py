import random
from from_scratch.utils.matrix import matrix_multiply, transpose
from from_scratch.optimizers.adam import Adam


class FeedForwardNetwork:
    def __init__(self,embedding_dimension: int,hidden_dimension: int,) -> None:
        if embedding_dimension <= 0:
            raise ValueError("Embedding dimension must be positive.")

        if hidden_dimension <= 0:
            raise ValueError("Hidden dimension must be positive.")

        self.embedding_dimension = embedding_dimension
        self.hidden_dimension = hidden_dimension

        self.first_weights = self._initialize_matrix(embedding_dimension,hidden_dimension)
        self.first_bias = [0.0] * hidden_dimension

        self.second_weights = self._initialize_matrix(hidden_dimension,embedding_dimension)
        self.second_bias = [0.0] * embedding_dimension

        self.optimizer = Adam()
        self.first_weights_first_moment = [[0.0 for _ in range(hidden_dimension)] for _ in range(embedding_dimension)]
        self.first_weights_second_moment = [[0.0 for _ in range(hidden_dimension)] for _ in range(embedding_dimension)]
        self.first_bias_first_moment = [0.0 for _ in range(hidden_dimension)]
        self.first_bias_second_moment = [0.0 for _ in range(hidden_dimension)]
        self.second_weights_first_moment = [[0.0 for _ in range(embedding_dimension)] for _ in range(hidden_dimension)]
        self.second_weights_second_moment = [[0.0 for _ in range(embedding_dimension)] for _ in range(hidden_dimension)]
        self.second_bias_first_moment = [0.0 for _ in range(embedding_dimension)]
        self.second_bias_second_moment = [0.0 for _ in range(embedding_dimension)]

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
        if not isinstance(inputs, list):
            raise TypeError("Inputs must be a list.")
        if not inputs:
            raise ValueError("Inputs cannot be empty.")
        for row in inputs:
            if not isinstance(row, list):
                raise TypeError("Every input row must be a list.")
            if len(row) != self.embedding_dimension:
                raise ValueError(f"Each input embedding must have {self.embedding_dimension} values.")
            if any(not isinstance(value, (int, float)) for value in row):
                raise TypeError("Every input value must be numeric.")

    def _add_bias(self,matrix: list[list[float]],bias: list[float]) -> list[list[float]]:
        return [[value + bias_value for value, bias_value in zip(row, bias)] for row in matrix]


    def _relu(self, matrix: list[list[float]]) -> list[list[float]]:
        return [[max(0.0, value) for value in row] for row in matrix]



    def forward(self,inputs: list[list[float]],) -> list[list[float]]:
        self._validate_inputs(inputs)
        self.last_inputs = [row.copy() for row in inputs]
        first_projection = matrix_multiply(inputs, self.first_weights)
        self.last_first_projection = self._add_bias(first_projection, self.first_bias)
        self.last_activated = self._relu(self.last_first_projection)
        second_projection = matrix_multiply(self.last_activated,self.second_weights)
        self.last_output = self._add_bias(second_projection,self.second_bias)
        return self.last_output

    def _relu_backward(self,output_gradient: list[list[float]],relu_inputs: list[list[float]]) -> list[list[float]]:
        return [[gradient if input_value > 0.0 else 0.0 for gradient, input_value in zip(gradient_row, input_row)] for gradient_row, input_row in zip(output_gradient, relu_inputs)]

    def _linear_backward(self,output_gradient: list[list[float]],layer_inputs: list[list[float]],weights: list[list[float]]) -> tuple[list[list[float]], list[list[float]], list[float]]:
        input_gradient = matrix_multiply(output_gradient, transpose(weights))
        weights_gradient = matrix_multiply(transpose(layer_inputs), output_gradient)
        bias_gradient = [sum(row[column] for row in output_gradient) for column in range(len(output_gradient[0]))]

        return input_gradient, weights_gradient, bias_gradient

    def _validate_output_gradient(self,output_gradient: list[list[float]]) -> None:
        if self.last_output is None:
            raise RuntimeError("forward() must be called before backward().")

        if not isinstance(output_gradient, list):
            raise TypeError("Output gradient must be a list.")
        if not output_gradient:
            raise ValueError("Output gradient cannot be empty.")

        if self.last_output is None:
            raise RuntimeError("Forward must be called before backward.")

        if len(output_gradient) != len(self.last_output):
            raise ValueError("Output gradient must match the output sequence length.")

        for row in output_gradient:
            if not isinstance(row, list):
                raise TypeError("Every output-gradient row must be a list.")
            if len(row) != self.embedding_dimension:
                raise ValueError(f"Each output-gradient row must have {self.embedding_dimension} values.")
            if any(not isinstance(value, (int, float)) for value in row):
                raise TypeError("Every gradient value must be numeric.")


    def backward(self, output_gradient: list[list[float]]) -> list[list[float]]:
        self._validate_output_gradient(output_gradient)
        if self.last_inputs is None or self.last_first_projection is None or self.last_activated is None:
            raise RuntimeError("Required forward values are missing.")

        activated_gradient, self.second_weights_gradient, self.second_bias_gradient = (self._linear_backward(output_gradient,self.last_activated,self.second_weights))

        first_projection_gradient = self._relu_backward(activated_gradient,self.last_first_projection)

        input_gradient, self.first_weights_gradient, self.first_bias_gradient = (self._linear_backward(first_projection_gradient,self.last_inputs,self.first_weights,))

        return input_gradient

    def update_parameters(self, learning_rate: float) -> None:
        if learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive.")
        if self.first_weights_gradient is None or self.first_bias_gradient is None or self.second_weights_gradient is None or self.second_bias_gradient is None:
            raise RuntimeError("backward() must be called before update_parameters().")

        self.optimizer.learning_rate = learning_rate
        self.optimizer.start_step()

        (
            self.first_weights,
            self.first_weights_first_moment,
            self.first_weights_second_moment,
        ) = self.optimizer.update(
            parameter=self.first_weights,
            gradient=self.first_weights_gradient,
            first_moment=self.first_weights_first_moment,
            second_moment=self.first_weights_second_moment,
        )

        (
            self.first_bias,
            self.first_bias_first_moment,
            self.first_bias_second_moment,
        ) = self.optimizer.update(
            parameter=self.first_bias,
            gradient=self.first_bias_gradient,
            first_moment=self.first_bias_first_moment,
            second_moment=self.first_bias_second_moment,
        )

        (
            self.second_weights,
            self.second_weights_first_moment,
            self.second_weights_second_moment,
        ) = self.optimizer.update(
            parameter=self.second_weights,
            gradient=self.second_weights_gradient,
            first_moment=self.second_weights_first_moment,
            second_moment=self.second_weights_second_moment,
        )

        (
            self.second_bias,
            self.second_bias_first_moment,
            self.second_bias_second_moment,
        ) = self.optimizer.update(
            parameter=self.second_bias,
            gradient=self.second_bias_gradient,
            first_moment=self.second_bias_first_moment,
            second_moment=self.second_bias_second_moment,
        )

        self.first_weights_gradient = None
        self.first_bias_gradient = None
        self.second_weights_gradient = None
        self.second_bias_gradient = None