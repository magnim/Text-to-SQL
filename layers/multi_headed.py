import math
import random

from layers.self_attention import SelfAttention


class MultiHeadSelfAttention:
    """A trainable multi-head self-attention layer.

    Args:
        embedding_dim: Number of features in each input token embedding.
        num_heads: Number of independent self-attention heads.

    Input shape:
        ``sequence_length x embedding_dim``

    Output shape:
        ``sequence_length x embedding_dim``
    """

    def __init__(self, embedding_dim: int, num_heads: int):
        if not isinstance(embedding_dim, int):
            raise TypeError("embedding_dim must be an integer.")

        if not isinstance(num_heads, int):
            raise TypeError("num_heads must be an integer.")

        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be greater than zero.")

        if num_heads <= 0:
            raise ValueError("num_heads must be greater than zero.")

        if embedding_dim % num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads.")

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads

        self.heads = [SelfAttention(embedding_dim=self.embedding_dim,attention_dim=self.head_dim) for _ in range(self.num_heads)]

        self.output_projection = self._initialize_output_projection()
        self.output_bias = [0.0 for _ in range(self.embedding_dim)]

        self.output_projection_gradient = None
        self.output_bias_gradient = None
        self.last_concatenated_output = None
        self.last_output = None

    def _initialize_output_projection(self) -> list[list[float]]:
        """Initialize the output projection with Xavier uniform values."""
        limit = math.sqrt(6.0 / (2 * self.embedding_dim))

        return [
            [
                random.uniform(-limit, limit)
                for _ in range(self.embedding_dim)
            ]
            for _ in range(self.embedding_dim)
        ]

    def _validate_embeddings(
        self,
        embeddings: list[list[float]],
    ) -> None:
        if not isinstance(embeddings, list):
            raise TypeError(
                "embeddings must be a list of token embeddings."
            )

        if not embeddings:
            raise ValueError(
                "embeddings must contain at least one token."
            )

        for token_index, token_embedding in enumerate(embeddings):
            if not isinstance(token_embedding, list):
                raise TypeError(f"Token embedding at index {token_index} ""must be a list.")

            if len(token_embedding) != self.embedding_dim:
                raise ValueError(f"Token embedding at index {token_index} must contain "f"{self.embedding_dim} features, but received "f"{len(token_embedding)}.")

            for feature_index, value in enumerate(token_embedding):
                if not isinstance(value, (int, float)):
                    raise TypeError(f"Feature {feature_index} of token {token_index} ""must be numeric.")

    def _validate_output_gradient(self,output_gradient: list[list[float]],) -> None:
        if self.last_output is None:
            raise RuntimeError("forward() must be called before backward().")

        if not isinstance(output_gradient, list):
            raise TypeError("output_gradient must be a list.")

        if not output_gradient:
            raise ValueError("output_gradient cannot be empty.")

        if len(output_gradient) != len(self.last_output):
            raise ValueError("output_gradient has an incorrect number of rows.")

        for token_index, gradient_row in enumerate(output_gradient):
            if not isinstance(gradient_row, list):
                raise TypeError(f"Gradient row {token_index} must be a list.")

            if len(gradient_row) != self.embedding_dim:
                raise ValueError(f"Gradient row {token_index} must contain {self.embedding_dim} features, but received {len(gradient_row)}.")

            for feature_index, value in enumerate(gradient_row):
                if not isinstance(value, (int, float)):
                    raise TypeError(f"Gradient feature {feature_index} of token {token_index} must be numeric.")

    def _validate_head_output(self,head_output: list[list[float]],head_index: int,sequence_length: int,) -> None:
        if not isinstance(head_output, list):
            raise TypeError(f"Head {head_index} output must be a list.")

        if len(head_output) != sequence_length:
            raise ValueError(f"Head {head_index} returned {len(head_output)} rows, but expected {sequence_length}.")

        for token_index, output_row in enumerate(head_output):
            if not isinstance(output_row, list):
                raise TypeError(f"Head {head_index}, token {token_index} output must be a list.")

            if len(output_row) != self.head_dim:
                raise ValueError(f"Head {head_index}, token {token_index} returned {len(output_row)} features, but expected {self.head_dim}.")

    def _concatenate_head_outputs(self,head_outputs: list[list[list[float]]],) -> list[list[float]]:
        if len(head_outputs) != self.num_heads:
            raise ValueError(f"Expected {self.num_heads} head outputs, but received {len(head_outputs)}.")

        sequence_length = len(head_outputs[0])
        concatenated_output = []

        for token_index in range(sequence_length):
            combined_token_features = []
            for head_output in head_outputs:
                combined_token_features.extend(head_output[token_index])
            if len(combined_token_features) != self.embedding_dim:
                raise RuntimeError(f"Concatenated output for token {token_index} contains {len(combined_token_features)} features, but expected {self.embedding_dim}.")
            concatenated_output.append(combined_token_features)

        return concatenated_output

    def _apply_output_projection(self,concatenated_output: list[list[float]],) -> list[list[float]]:
        final_output = []

        for token_row in concatenated_output:
            projected_row = []

            for output_feature in range(self.embedding_dim):
                projected_value = self.output_bias[output_feature]

                for input_feature in range(self.embedding_dim):
                    projected_value += (token_row[input_feature]* self.output_projection[input_feature][output_feature])

                projected_row.append(projected_value)

            final_output.append(projected_row)

        return final_output

    def forward(self,embeddings: list[list[float]],) -> list[list[float]]:
        """Run all heads, concatenate their outputs, and project them."""
        self._validate_embeddings(embeddings)
        sequence_length = len(embeddings)
        head_outputs = []

        for head_index, head in enumerate(self.heads):
            head_output = head.forward(embeddings)
            self._validate_head_output(
                head_output,
                head_index,
                sequence_length,
            )
            head_outputs.append(head_output)

        concatenated_output = self._concatenate_head_outputs(
            head_outputs
        )
        self.last_concatenated_output = concatenated_output
        self.last_output = self._apply_output_projection(
            concatenated_output
        )

        return self.last_output

    def _output_projection_backward(self,output_gradient: list[list[float]],) -> list[list[float]]:
        if self.last_concatenated_output is None:
            raise RuntimeError(
                "forward() must be called before backward()."
            )

        sequence_length = len(output_gradient)
        self.output_projection_gradient = [
            [0.0 for _ in range(self.embedding_dim)]
            for _ in range(self.embedding_dim)
        ]
        self.output_bias_gradient = [
            0.0 for _ in range(self.embedding_dim)
        ]
        concatenated_gradient = [
            [0.0 for _ in range(self.embedding_dim)]
            for _ in range(sequence_length)
        ]

        for token_index in range(sequence_length):
            concatenated_row = self.last_concatenated_output[
                token_index
            ]
            gradient_row = output_gradient[token_index]

            for output_feature in range(self.embedding_dim):
                gradient_value = gradient_row[output_feature]
                self.output_bias_gradient[output_feature] += (
                    gradient_value
                )

                for input_feature in range(self.embedding_dim):
                    self.output_projection_gradient[
                        input_feature
                    ][output_feature] += (
                        concatenated_row[input_feature]
                        * gradient_value
                    )

                    concatenated_gradient[
                        token_index
                    ][input_feature] += (
                        gradient_value
                        * self.output_projection[
                            input_feature
                        ][output_feature]
                    )

        return concatenated_gradient

    def _split_gradient(self,concatenated_gradient: list[list[float]],) -> list[list[list[float]]]:
        if not isinstance(concatenated_gradient, list):
            raise TypeError(
                "concatenated_gradient must be a list."
            )

        if not concatenated_gradient:
            raise ValueError(
                "concatenated_gradient cannot be empty."
            )

        for token_index, gradient_row in enumerate(
            concatenated_gradient
        ):
            if not isinstance(gradient_row, list):
                raise TypeError(
                    f"Gradient row {token_index} must be a list."
                )

            if len(gradient_row) != self.embedding_dim:
                raise ValueError(
                    f"Gradient row {token_index} must contain "
                    f"{self.embedding_dim} features, but received "
                    f"{len(gradient_row)}."
                )

        head_gradients = []

        for head_index in range(self.num_heads):
            start_feature = head_index * self.head_dim
            end_feature = start_feature + self.head_dim
            head_gradient = [
                gradient_row[start_feature:end_feature]
                for gradient_row in concatenated_gradient
            ]
            head_gradients.append(head_gradient)

        return head_gradients

    def backward(self,output_gradient: list[list[float]],) -> list[list[float]]:
        """Backpropagate through projection and all attention heads."""
        self._validate_output_gradient(output_gradient)
        sequence_length = len(output_gradient)

        concatenated_gradient = self._output_projection_backward(
            output_gradient
        )
        head_gradients = self._split_gradient(
            concatenated_gradient
        )
        final_input_gradient = [
            [0.0 for _ in range(self.embedding_dim)]
            for _ in range(sequence_length)
        ]

        for head_index, (head, head_gradient) in enumerate(
            zip(self.heads, head_gradients)
        ):
            input_gradient = head.backward(head_gradient)

            if len(input_gradient) != sequence_length:
                raise ValueError(
                    f"Head {head_index} returned an invalid input-"
                    "gradient sequence length."
                )

            for token_index, gradient_row in enumerate(
                input_gradient
            ):
                if len(gradient_row) != self.embedding_dim:
                    raise ValueError(
                        f"Head {head_index}, token {token_index} "
                        "returned an invalid input-gradient width."
                    )

                for feature_index in range(self.embedding_dim):
                    final_input_gradient[
                        token_index
                    ][feature_index] += gradient_row[feature_index]

        return final_input_gradient

    def update_parameters(self, learning_rate: float) -> None:
        """Apply gradient descent to every head and output projection."""
        if not isinstance(learning_rate, (int, float)):
            raise TypeError("learning_rate must be numeric.")

        if learning_rate < 0:
            raise ValueError(
                "learning_rate cannot be negative."
            )

        if (
            self.output_projection_gradient is None
            or self.output_bias_gradient is None
        ):
            raise RuntimeError(
                "backward() must be called before "
                "update_parameters()."
            )

        for head in self.heads:
            head.update_parameters(learning_rate)

        for input_feature in range(self.embedding_dim):
            for output_feature in range(self.embedding_dim):
                self.output_projection[
                    input_feature
                ][output_feature] -= (
                    learning_rate
                    * self.output_projection_gradient[
                        input_feature
                    ][output_feature]
                )

        for feature_index in range(self.embedding_dim):
            self.output_bias[feature_index] -= (
                learning_rate
                * self.output_bias_gradient[feature_index]
            )


if __name__ == "__main__":
    random.seed(42)

    attention = MultiHeadSelfAttention(
        embedding_dim=4,
        num_heads=2,
    )

    sample_embeddings = [
        [0.1, 0.2, 0.3, 0.4],
        [0.5, 0.6, 0.7, 0.8],
        [0.9, 1.0, 1.1, 1.2],
    ]

    attention_output = attention.forward(sample_embeddings)

    print("Multi-head attention output:")
    for output_row in attention_output:
        print(output_row)

    sample_output_gradient = [
        [1.0, 1.0, 1.0, 1.0]
        for _ in sample_embeddings
    ]

    input_gradient = attention.backward(
        sample_output_gradient
    )

    print("\nInput gradient:")
    for gradient_row in input_gradient:
        print(gradient_row)

    attention.update_parameters(learning_rate=0.01)
    print("\nParameters updated successfully.")