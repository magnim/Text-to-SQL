class ResidualConnection:
    def forward(self,inputs: list[list[float]],layer_output: list[list[float]]) -> list[list[float]]:
        self._validate_matrices(inputs, layer_output)

        return [
            [
                input_value + output_value
                for input_value, output_value in zip(input_row, output_row)
            ]
            for input_row, output_row in zip(inputs, layer_output)
        ]

    def backward(self,output_gradient: list[list[float]]) -> tuple[list[list[float]], list[list[float]]]:
        self._validate_gradient(output_gradient)

        input_gradient = [row.copy() for row in output_gradient]

        layer_output_gradient = [row.copy() for row in output_gradient]

        return input_gradient, layer_output_gradient

    def _validate_matrices(self,inputs: list[list[float]],layer_output: list[list[float]]) -> None:
        if not inputs:
            raise ValueError("Inputs cannot be empty.")

        if not layer_output:
            raise ValueError("Layer output cannot be empty.")

        if len(inputs) != len(layer_output):
            raise ValueError("Inputs and layer output must have the same number of rows.")

        expected_columns = len(inputs[0])

        if expected_columns == 0:
            raise ValueError("Input rows cannot be empty.")

        for input_row, output_row in zip(inputs, layer_output):
            if len(input_row) != expected_columns:
                raise ValueError("All input rows must have the same number of columns.")

            if len(output_row) != expected_columns:
                raise ValueError("Inputs and layer output must have identical shapes.")

    def _validate_gradient(self,output_gradient: list[list[float]],) -> None:
        if not output_gradient:
            raise ValueError("Output gradient cannot be empty.")

        expected_columns = len(output_gradient[0])

        if expected_columns == 0:
            raise ValueError("Output-gradient rows cannot be empty.")

        for row in output_gradient:
            if len(row) != expected_columns:
                raise ValueError("All output-gradient rows must have the same length.")