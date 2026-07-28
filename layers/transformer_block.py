from layers.feed_forward import FeedForwardNetwork
from layers.layer_normalization import LayerNormalization
from layers.multi_headed import MultiHeadSelfAttention
from layers.residual_connection import ResidualConnection

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
            if len(row) != self.embedding_dimension:
                raise ValueError(f"Each input row must contain {self.embedding_dimension} values.")

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

