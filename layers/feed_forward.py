import random
from utils.matrix import matrix_multiply


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



    def forward(
            self,
            inputs: list[list[float]],
    ) -> list[list[float]]:
        self._validate_inputs(inputs)
        self.last_inputs = inputs
        first_projection = matrix_multiply(inputs, self.first_weights)
        self.last_first_projection = self._add_bias(first_projection, self.first_bias)
        self.last_activated = self._relu(self.last_first_projection)
        second_projection = matrix_multiply(self.last_activated,self.second_weights,)
        self.last_output = self._add_bias(second_projection,self.second_bias,)
        return self.last_output