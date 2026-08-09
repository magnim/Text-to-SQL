import random


class SchemaRoleEmbedding:

    NORMAL = 0
    TABLE = 1
    COLUMN = 2

    def __init__(self,embedding_dimension: int) -> None:

        if embedding_dimension <= 0:
            raise ValueError("embedding_dimension must be positive.")

        self.embedding_dimension = embedding_dimension
        self.number_of_roles = 3
        self.embedding_matrix = [[random.uniform(-0.1, 0.1) for _ in range(embedding_dimension)] for _ in range(self.number_of_roles)]
        self.last_role_ids: list[int] | None = None
        self.gradients = [[0.0 for _ in range(embedding_dimension)] for _ in range(self.number_of_roles)]

    def forward(self,role_ids: list[int]) -> list[list[float]]:
        role_embeddings = []
        for role_id in role_ids:
            if not isinstance(role_id, int):
                raise TypeError("Every role ID must be an integer.")
            if role_id < 0 or role_id >= self.number_of_roles:
                raise ValueError(f"Invalid schema role ID: {role_id}.")
            role_embeddings.append(self.embedding_matrix[role_id].copy())
        self.last_role_ids = role_ids.copy()
        return role_embeddings

    def backward(self,output_gradients: list[list[float]]) -> None:

        if self.last_role_ids is None:
            raise RuntimeError("forward() must be called before backward().")

        if len(output_gradients) != len(self.last_role_ids):
            raise ValueError("output_gradients must have the same sequence length as the last forward pass.")

        # Reset gradients from the previous backward pass.
        self.gradients = [[0.0 for _ in range(self.embedding_dimension)] for _ in range(self.number_of_roles)]

        for role_id, gradient_row in zip(self.last_role_ids,output_gradients):
            if len(gradient_row) != self.embedding_dimension:
                raise ValueError("Gradient row has incorrect dimension.")

            for dimension_index in range(self.embedding_dimension):
                self.gradients[role_id][dimension_index] += (gradient_row[dimension_index])

    def update_parameters(self,learning_rate: float) -> None:

        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")

        for role_id in range(self.number_of_roles):
            for dimension_index in range(self.embedding_dimension):
                self.embedding_matrix[role_id][dimension_index] -= (learning_rate * self.gradients[role_id][dimension_index])

