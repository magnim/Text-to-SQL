import math
import random

from utils.matrix import matrix_add, matrix_multiply, transpose


class SelfAttention:
    def __init__(self, embedding_dim: int, attention_dim: int):
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be greater than 0.")

        if attention_dim <= 0:
            raise ValueError("attention_dim must be greater than 0.")

        self.embedding_dim = embedding_dim
        self.attention_dim = attention_dim

        self.query_projection = self._initialize_weights()
        self.key_projection = self._initialize_weights()
        self.value_projection = self._initialize_weights()

        self.query_bias = self._initialize_bias()
        self.key_bias = self._initialize_bias()
        self.value_bias = self._initialize_bias()

        self.query_projection_gradient = None
        self.key_projection_gradient = None
        self.value_projection_gradient = None

        self.query_bias_gradient = None
        self.key_bias_gradient = None
        self.value_bias_gradient = None

        self.last_embeddings = None
        self.last_queries = None
        self.last_keys = None
        self.last_values = None
        self.last_attention_scores = None
        self.last_scaled_attention_scores = None
        self.last_attention_weights = None
        self.last_output = None

    def _initialize_weights(self) -> list[list[float]]:
        limit = math.sqrt(6.0 / (self.embedding_dim + self.attention_dim))

        return [
            [random.uniform(-limit, limit) for _ in range(self.attention_dim)]
            for _ in range(self.embedding_dim)
        ]

    def _initialize_bias(self) -> list[float]:
        return [0.0 for _ in range(self.attention_dim)]

    def _validate_embeddings(self, embeddings: list[list[float]]) -> None:
        if not embeddings:
            raise ValueError("Embeddings cannot be empty.")

        for row in embeddings:
            if len(row) != self.embedding_dim:
                raise ValueError(
                    f"Each embedding must have {self.embedding_dim} values, "
                    f"but received {len(row)}."
                )

    def _validate_output_gradient(self, output_gradient: list[list[float]]) -> None:
        if self.last_output is None:
            raise RuntimeError("Forward must be called before backward.")

        if len(output_gradient) != len(self.last_output):
            raise ValueError("Output gradient has an incorrect number of rows.")

        for row in output_gradient:
            if len(row) != self.attention_dim:
                raise ValueError(
                    f"Each output-gradient row must have {self.attention_dim} values."
                )

    def _add_bias(self, matrix: list[list[float]], bias: list[float]) -> list[list[float]]:
        return [
            [row[index] + bias[index] for index in range(len(row))]
            for row in matrix
        ]

    def _softmax(self, attention_scores: list[list[float]]) -> list[list[float]]:
        attention_weights = []

        for row in attention_scores:
            maximum_score = max(row)
            exponentials = [math.exp(score - maximum_score) for score in row]
            total = sum(exponentials)
            attention_weights.append([value / total for value in exponentials])

        return attention_weights

    def _scale_attention_scores(self, attention_scores: list[list[float]]) -> list[list[float]]:
        scale = math.sqrt(self.attention_dim)
        return [[score / scale for score in row] for row in attention_scores]

    def _attention_output(
        self,
        attention_weights: list[list[float]],
        values: list[list[float]],
    ) -> list[list[float]]:
        self.last_attention_weights = attention_weights
        self.last_values = values
        self.last_output = matrix_multiply(attention_weights, values)

        return self.last_output

    def forward(self, embeddings: list[list[float]]) -> list[list[float]]:
        self._validate_embeddings(embeddings)
        self.last_embeddings = embeddings

        queries = matrix_multiply(embeddings, self.query_projection)
        self.last_queries = self._add_bias(queries, self.query_bias)

        keys = matrix_multiply(embeddings, self.key_projection)
        self.last_keys = self._add_bias(keys, self.key_bias)

        values = matrix_multiply(embeddings, self.value_projection)
        self.last_values = self._add_bias(values, self.value_bias)

        self.last_attention_scores = matrix_multiply(self.last_queries, transpose(self.last_keys))
        self.last_scaled_attention_scores = self._scale_attention_scores(self.last_attention_scores)
        attention_weights = self._softmax(self.last_scaled_attention_scores)

        return self._attention_output(attention_weights, self.last_values)

    def _attention_output_backward(
        self,
        output_gradient: list[list[float]],
    ) -> tuple[list[list[float]], list[list[float]]]:
        if self.last_attention_weights is None or self.last_values is None:
            raise RuntimeError("Forward must be called before backward.")

        attention_weights_gradient = matrix_multiply(output_gradient, transpose(self.last_values))
        values_gradient = matrix_multiply(transpose(self.last_attention_weights), output_gradient)

        return attention_weights_gradient, values_gradient

    def _softmax_backward(
        self,
        softmax_output: list[float],
        output_gradient: list[float],
    ) -> list[float]:
        input_gradient = [0.0 for _ in softmax_output]

        for input_index in range(len(softmax_output)):
            for output_index in range(len(softmax_output)):
                if input_index == output_index:
                    local_gradient = softmax_output[output_index] * (1.0 - softmax_output[output_index])
                else:
                    local_gradient = -softmax_output[output_index] * softmax_output[input_index]

                input_gradient[input_index] += output_gradient[output_index] * local_gradient

        return input_gradient

    def _attention_weights_backward(
        self,
        attention_weights_gradient: list[list[float]],
    ) -> list[list[float]]:
        if self.last_attention_weights is None:
            raise RuntimeError("Forward must be called before backward.")

        scaled_scores_gradient = []

        for row_index in range(len(self.last_attention_weights)):
            row_gradient = self._softmax_backward(
                self.last_attention_weights[row_index],
                attention_weights_gradient[row_index],
            )
            scaled_scores_gradient.append(row_gradient)

        return scaled_scores_gradient

    def _scaled_scores_backward(
        self,
        scaled_scores_gradient: list[list[float]],
    ) -> list[list[float]]:
        scale = math.sqrt(self.attention_dim)
        return [[gradient / scale for gradient in row] for row in scaled_scores_gradient]

    def _scores_backward(
        self,
        scores_gradient: list[list[float]],
    ) -> tuple[list[list[float]], list[list[float]]]:
        if self.last_queries is None or self.last_keys is None:
            raise RuntimeError("Forward must be called before backward.")

        queries_gradient = matrix_multiply(scores_gradient, self.last_keys)
        keys_gradient = matrix_multiply(transpose(scores_gradient), self.last_queries)

        return queries_gradient, keys_gradient

    def _projection_backward(
        self,
        output_gradient: list[list[float]],
        weights: list[list[float]],
    ) -> tuple[list[list[float]], list[list[float]], list[float]]:
        if self.last_embeddings is None:
            raise RuntimeError("Forward must be called before backward.")

        input_gradient = matrix_multiply(output_gradient, transpose(weights))
        weights_gradient = matrix_multiply(transpose(self.last_embeddings), output_gradient)
        bias_gradient = [0.0 for _ in range(self.attention_dim)]

        for row in output_gradient:
            for column_index in range(self.attention_dim):
                bias_gradient[column_index] += row[column_index]

        return input_gradient, weights_gradient, bias_gradient

    def backward(self, output_gradient: list[list[float]]) -> list[list[float]]:
        self._validate_output_gradient(output_gradient)

        attention_weights_gradient, values_gradient = self._attention_output_backward(output_gradient)
        scaled_scores_gradient = self._attention_weights_backward(attention_weights_gradient)
        scores_gradient = self._scaled_scores_backward(scaled_scores_gradient)
        queries_gradient, keys_gradient = self._scores_backward(scores_gradient)

        (
            x_gradient_from_query,
            self.query_projection_gradient,
            self.query_bias_gradient,
        ) = self._projection_backward(queries_gradient, self.query_projection)

        (
            x_gradient_from_key,
            self.key_projection_gradient,
            self.key_bias_gradient,
        ) = self._projection_backward(keys_gradient, self.key_projection)

        (
            x_gradient_from_value,
            self.value_projection_gradient,
            self.value_bias_gradient,
        ) = self._projection_backward(values_gradient, self.value_projection)

        return matrix_add(
            matrix_add(x_gradient_from_query, x_gradient_from_key),
            x_gradient_from_value,
        )

    def _update_matrix(
        self,
        matrix: list[list[float]],
        gradient: list[list[float]],
        learning_rate: float,
    ) -> None:
        for row_index in range(len(matrix)):
            for column_index in range(len(matrix[0])):
                matrix[row_index][column_index] -= learning_rate * gradient[row_index][column_index]

    def _update_vector(
        self,
        vector: list[float],
        gradient: list[float],
        learning_rate: float,
    ) -> None:
        for index in range(len(vector)):
            vector[index] -= learning_rate * gradient[index]

    def update_parameters(self, learning_rate: float) -> None:
        if learning_rate < 0:
            raise ValueError("learning_rate cannot be negative.")

        gradients = [
            self.query_projection_gradient,
            self.key_projection_gradient,
            self.value_projection_gradient,
            self.query_bias_gradient,
            self.key_bias_gradient,
            self.value_bias_gradient,
        ]

        if any(gradient is None for gradient in gradients):
            raise RuntimeError("Backward must be called before updating parameters.")

        self._update_matrix(self.query_projection, self.query_projection_gradient, learning_rate)
        self._update_matrix(self.key_projection, self.key_projection_gradient, learning_rate)
        self._update_matrix(self.value_projection, self.value_projection_gradient, learning_rate)

        self._update_vector(self.query_bias, self.query_bias_gradient, learning_rate)
        self._update_vector(self.key_bias, self.key_bias_gradient, learning_rate)
        self._update_vector(self.value_bias, self.value_bias_gradient, learning_rate)


if __name__ == "__main__":
    random.seed(42)

    attention = SelfAttention(embedding_dim=4, attention_dim=2)

    embeddings = [
        [0.1, 0.2, 0.3, 0.4],
        [0.5, 0.6, 0.7, 0.8],
        [0.9, 1.0, 1.1, 1.2],
    ]

    attention_output = attention.forward(embeddings)

    print("Attention output:")

    for row in attention_output:
        print(row)