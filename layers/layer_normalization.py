class LayerNormalization:
    def __init__(self,embedding_dimension: int,epsilon: float = 1e-5 ) -> None:
        if embedding_dimension <= 0:
            raise ValueError("Embedding dimension must be positive.")

        if epsilon <= 0.0:
            raise ValueError("Epsilon must be positive.")

        self.embedding_dimension = embedding_dimension
        self.epsilon = epsilon

        self.gamma = [1.0] * embedding_dimension
        self.beta = [0.0] * embedding_dimension

        self.last_inputs = None
        self.last_means = None
        self.last_variances = None
        self.last_inverse_standard_deviations = None
        self.last_normalized_inputs = None
        self.last_output = None

        self.gamma_gradient = None
        self.beta_gradient = None

    def _validate_inputs(self,inputs: list[list[float]]) -> None:
        if not inputs:
            raise ValueError("Inputs cannot be empty.")

        for row in inputs:
            if len(row) != self.embedding_dimension:
                raise ValueError(f"Each input row must contain {self.embedding_dimension} values.")

    def _calculate_mean(self,values: list[float]) -> float:
        return sum(values) / len(values)

    def _calculate_variance(self,values: list[float],mean: float) -> float:
        squared_differences = [(value - mean) ** 2 for value in values]
        return sum(squared_differences) / len(values)

    def _normalize_row(self,values: list[float],mean: float,variance: float) -> tuple[list[float], float]:
        inverse_standard_deviation = (variance + self.epsilon) ** -0.5

        normalized_values = [(value - mean) * inverse_standard_deviation for value in values]
        return normalized_values, inverse_standard_deviation

    def _scale_and_shift(self,normalized_values: list[float],) -> list[float]:
        return [gamma_value * normalized_value + beta_value
                for normalized_value, gamma_value, beta_value in zip(normalized_values,self.gamma,self.beta)]

    def forward(self,inputs: list[list[float]],) -> list[list[float]]:
        self._validate_inputs(inputs)
        self.last_inputs = inputs

        means = []
        variances = []
        inverse_standard_deviations = []
        normalized_inputs = []
        outputs = []

        for row in inputs:
            mean = self._calculate_mean(row)
            variance = self._calculate_variance(row, mean)

            normalized_row, inverse_standard_deviation = (self._normalize_row(row,mean,variance))

            output_row = self._scale_and_shift(normalized_row)

            means.append(mean)
            variances.append(variance)
            inverse_standard_deviations.append(inverse_standard_deviation)
            normalized_inputs.append(normalized_row)
            outputs.append(output_row)

        self.last_means = means
        self.last_variances = variances
        self.last_inverse_standard_deviations = (inverse_standard_deviations)
        self.last_normalized_inputs = normalized_inputs
        self.last_output = outputs

        return outputs

    def _calculate_beta_gradient(self,output_gradient: list[list[float]],) -> list[float]:
        return [sum(row[column] for row in output_gradient) for column in range(self.embedding_dimension)]

    def _calculate_gamma_gradient(self,output_gradient: list[list[float]]) -> list[float]:
        return [sum(gradient_row[column]* normalized_row[column]
                    for gradient_row, normalized_row in zip(output_gradient,self.last_normalized_inputs,))
                for column in range(self.embedding_dimension)]

    def _normalize_backward(self,output_gradient: list[list[float]],) -> list[list[float]]:
        input_gradients = []

        for (gradient_row,normalized_row,inverse_standard_deviation) in zip(output_gradient,self.last_normalized_inputs,self.last_inverse_standard_deviations):
            feature_count = self.embedding_dimension
            sum_gradient = sum(gradient_row)
            sum_gradient_times_normalized = sum(gradient * normalized for gradient, normalized in zip(gradient_row,normalized_row,))
            input_gradient = []
            for (gradient,normalized,gamma) in zip(gradient_row,normalized_row,self.gamma):
                value = (gamma * inverse_standard_deviation / feature_count) * (feature_count * gradient- sum_gradient- normalized* sum_gradient_times_normalized)
                input_gradient.append(value)
            input_gradients.append(input_gradient)

        return input_gradients

    def _validate_output_gradient(self,output_gradient: list[list[float]]) -> None:
        if not output_gradient:
            raise ValueError("Output gradient cannot be empty.")

        if len(output_gradient) != len(self.last_output):
            raise ValueError("Gradient shape does not match output.")

        for row in output_gradient:
            if len(row) != self.embedding_dimension:
                raise ValueError(f"Each gradient row must contain {self.embedding_dimension} values.")

    def update_parameters(self,learning_rate: float) -> None:
        if learning_rate <= 0.0:
            raise ValueError("Learning rate must be positive.")

        for index in range(self.embedding_dimension):
            self.gamma[index] -= (learning_rate * self.gamma_gradient[index])
            self.beta[index] -= (learning_rate * self.beta_gradient[index])