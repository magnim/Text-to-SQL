from from_scratch.layers.feed_forward import FeedForwardNetwork
from from_scratch.layers.layer_normalization import LayerNormalization
from from_scratch.layers.multi_headed import MultiHeadSelfAttention
from from_scratch.layers.residual_connection import ResidualConnection

class TransformerBlock:
    def __init__(self,embedding_dimension: int,number_of_heads: int,hidden_dimension: int) -> None:
        self._validate_dimensions(embedding_dimension,number_of_heads,hidden_dimension)

        self.embedding_dimension = embedding_dimension
        self.number_of_heads = number_of_heads
        self.hidden_dimension = hidden_dimension

        self.attention = MultiHeadSelfAttention(embedding_dim=embedding_dimension,num_heads=number_of_heads)

        self.attention_residual = ResidualConnection()

        self.first_layer_norm = LayerNormalization(embedding_dimension=embedding_dimension)

        self.feed_forward = FeedForwardNetwork(embedding_dimension=embedding_dimension,hidden_dimension=hidden_dimension)

        self.feed_forward_residual = ResidualConnection()

        self.second_layer_norm = LayerNormalization(embedding_dimension=embedding_dimension)

        self.last_inputs = None
        self.last_attention_output = None
        self.last_attention_residual_output = None
        self.last_first_normalized_output = None
        self.last_feed_forward_output = None
        self.last_feed_forward_residual_output = None
        self.last_output = None

    def _validate_dimensions(self,embedding_dimension: int,number_of_heads: int,hidden_dimension: int) -> None:
        if embedding_dimension <= 0:
            raise ValueError("Embedding dimension must be positive.")

        if number_of_heads <= 0:
            raise ValueError("Number of heads must be positive.")

        if hidden_dimension <= 0:
            raise ValueError("Hidden dimension must be positive.")

        if embedding_dimension % number_of_heads != 0:
            raise ValueError("Embedding dimension must be divisible by the number of attention heads.")

    def _validate_inputs(self,inputs: list[list[float]]) -> None:
        if not inputs:
            raise ValueError("Inputs cannot be empty.")
        for row in inputs:
            if not isinstance(row, list):
                raise TypeError("Every input row must be a list.")
            if len(row) != self.embedding_dimension:
                raise ValueError(f"Each input row must contain {self.embedding_dimension} values.")
            if any(not isinstance(value, (int, float)) for value in row):
                raise TypeError("Every input value must be numeric.")

    def _add_gradients(self,first_gradient: list[list[float]],second_gradient: list[list[float]]) -> list[list[float]]:
        if len(first_gradient) != len(second_gradient):
            raise ValueError("Gradient matrices must have the same number of rows.")
        combined_gradient = []
        for first_row, second_row in zip(first_gradient,second_gradient):
            if len(first_row) != len(second_row):
                raise ValueError("Gradient matrices must have identical shapes.")
            combined_row = [first_value + second_value for first_value, second_value in zip(first_row,second_row)]
            combined_gradient.append(combined_row)
        return combined_gradient

    def _validate_output_gradient(self,output_gradient: list[list[float]]) -> None:
        if self.last_output is None:
            raise RuntimeError("forward() must be called before backward().")

        if not output_gradient:
            raise ValueError("Output gradient cannot be empty.")
        if not isinstance(output_gradient, list):
            raise TypeError("Output gradient must be a list.")

        if len(output_gradient) != len(self.last_output):
            raise ValueError("Output gradient must have the same number of rows as the block output.")

        for gradient_row, output_row in zip(output_gradient,self.last_output):
            if len(gradient_row) != len(output_row):
                raise ValueError("Output gradient must have the same shape as the block output.")
            if not isinstance(gradient_row, list):
                raise TypeError("Every gradient row must be a list.")

            if any(not isinstance(value, (int, float)) for value in gradient_row):
                raise TypeError("Every gradient value must be numeric.")

    def forward(self,inputs: list[list[float]]) -> list[list[float]]:
        self._validate_inputs(inputs)
        self.last_inputs = [row.copy() for row in inputs]
        attention_output = self.attention.forward(inputs)
        self.last_attention_output = attention_output
        attention_residual_output = (self.attention_residual.forward(inputs,attention_output))
        self.last_attention_residual_output = attention_residual_output
        first_normalized_output = (self.first_layer_norm.forward(attention_residual_output))
        self.last_first_normalized_output = first_normalized_output
        feed_forward_output = self.feed_forward.forward(first_normalized_output)
        self.last_feed_forward_output = feed_forward_output
        feed_forward_residual_output = (self.feed_forward_residual.forward(first_normalized_output,feed_forward_output))
        self.last_feed_forward_residual_output = feed_forward_residual_output
        output = self.second_layer_norm.forward(feed_forward_residual_output)
        self.last_output = output
        return output

    def backward(self,output_gradient: list[list[float]]) -> list[list[float]]:
        self._validate_output_gradient(output_gradient)
        feed_forward_residual_gradient = (self.second_layer_norm.backward(output_gradient))
        (direct_first_normalized_gradient,feed_forward_output_gradient) = self.feed_forward_residual.backward(feed_forward_residual_gradient)
        feed_forward_input_gradient = (self.feed_forward.backward(feed_forward_output_gradient))
        first_normalized_gradient = self._add_gradients(direct_first_normalized_gradient,feed_forward_input_gradient,)
        attention_residual_gradient = (self.first_layer_norm.backward(first_normalized_gradient))
        (direct_input_gradient,attention_output_gradient) = self.attention_residual.backward(attention_residual_gradient)
        attention_input_gradient = (self.attention.backward(attention_output_gradient))
        input_gradient = self._add_gradients(direct_input_gradient,attention_input_gradient)
        return input_gradient

    def update_parameters(self, learning_rate: float) -> None:
        if learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive.")

        self.attention.update_parameters(learning_rate)
        self.first_layer_norm.update_parameters(learning_rate)
        self.feed_forward.update_parameters(learning_rate)
        self.second_layer_norm.update_parameters(learning_rate)
