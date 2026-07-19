import math

from layers.self_attention import SelfAttention


def matrix_shape(matrix: list[list[float]]) -> tuple[int, int]:
    if not matrix:
        return 0, 0

    return len(matrix), len(matrix[0])


def validate_matrix(
    matrix: list[list[float]],
    expected_shape: tuple[int, int],
    matrix_name: str,
) -> None:
    actual_shape = matrix_shape(matrix)

    if actual_shape != expected_shape:
        raise ValueError(f"{matrix_name} has shape {actual_shape}, but expected {expected_shape}.")

    for row_index, row in enumerate(matrix):
        for column_index, value in enumerate(row):
            if not math.isfinite(value):
                raise ValueError(f"{matrix_name}[{row_index}][{column_index}] is not finite: {value}")


def mse_forward(prediction: list[list[float]],target: list[list[float]]) -> float:
    prediction_shape = matrix_shape(prediction)
    target_shape = matrix_shape(target)

    if prediction_shape != target_shape:
        raise ValueError(f"Prediction shape {prediction_shape} does not match target shape {target_shape}.")

    total_squared_error = 0.0
    number_of_values = 0

    for row_index in range(len(prediction)):
        for column_index in range(len(prediction[0])):
            difference = (prediction[row_index][column_index]- target[row_index][column_index])

            total_squared_error += difference ** 2
            number_of_values += 1

    if number_of_values == 0:
        raise ValueError("Cannot calculate MSE for an empty matrix.")

    return total_squared_error / number_of_values


def mse_backward(prediction: list[list[float]],target: list[list[float]]) -> list[list[float]]:
    prediction_shape = matrix_shape(prediction)
    target_shape = matrix_shape(target)

    if prediction_shape != target_shape:
        raise ValueError(f"Prediction shape {prediction_shape} does not match target shape {target_shape}.")

    number_of_values = len(prediction) * len(prediction[0])

    return [
        [2.0 * (prediction[row_index][column_index]- target[row_index][column_index])/ number_of_values
            for column_index in range(len(prediction[0]))
        ]
        for row_index in range(len(prediction))
    ]


def train_self_attention() -> None:
    embedding_dimension = 4
    attention_dimension = 2
    sequence_length = 3
    learning_rate = 0.01
    number_of_epochs = 200

    attention = SelfAttention(
        embedding_dim=embedding_dimension,
        attention_dim=attention_dimension,
    )

    inputs = [
        [0.2, -0.1, 0.4, 0.7],
        [0.8, 0.3, -0.5, 0.1],
        [-0.4, 0.9, 0.6, -0.2],
    ]

    target = [
        [0.5, -0.2],
        [0.1, 0.7],
        [-0.3, 0.4],
    ]

    expected_input_shape = (sequence_length,embedding_dimension)

    expected_output_shape = (sequence_length,attention_dimension)

    validate_matrix(inputs,expected_input_shape,"Inputs",)

    validate_matrix(target,expected_output_shape,"Target",)

    initial_loss = None
    final_loss = None

    print("Training single-head self-attention")
    print(f"Input shape:  {expected_input_shape}")
    print(f"Output shape: {expected_output_shape}")
    print()

    for epoch in range(1, number_of_epochs + 1):
        prediction = attention.forward(inputs)

        validate_matrix(prediction,expected_output_shape,"Prediction")

        loss = mse_forward(prediction, target)

        output_gradient = mse_backward(prediction,target)

        validate_matrix(output_gradient,expected_output_shape,"Output gradient",)

        input_gradient = attention.backward(output_gradient)

        validate_matrix(input_gradient,expected_input_shape,"Input gradient")

        attention.update_parameters(learning_rate,)

        if initial_loss is None:
            initial_loss = loss

        final_loss = loss

        if epoch == 1 or epoch % 20 == 0:
            print(f"Epoch {epoch:3d} | "f"Loss: {loss:.8f}")

    if initial_loss is None or final_loss is None:
        raise RuntimeError("Training did not run.")

    print()
    print(f"Initial loss: {initial_loss:.8f}")
    print(f"Final loss:   {final_loss:.8f}")

    if final_loss >= initial_loss:
        raise AssertionError("Training failed: final loss was not lower ""than the initial loss.")

    final_prediction = attention.forward(inputs)

    print()
    print("Final prediction:")

    for row in final_prediction:
        print([round(value, 4) for value in row])

    print()
    print("Target:")

    for row in target:
        print(row)

    print()
    print("Self-attention training test passed.")


if __name__ == "__main__":
    train_self_attention()