def dot_product(vector_a: list[float], vector_b: list[float]) -> float:
    if len(vector_a) != len(vector_b):
        raise ValueError("Vectors must have the same length")

    result = 0.0

    for index in range(len(vector_a)):
        result += vector_a[index] * vector_b[index]

    return result

def transpose(matrix: list[list[float]]) -> list[list[float]]:
    if not matrix or not matrix[0]:
        raise ValueError("Matrix cannot be empty")

    row_count = len(matrix)
    column_count = len(matrix[0])

    if any(len(row) != column_count for row in matrix):
        raise ValueError("Matrix rows must have the same length")

    transposed = []

    for column in range(column_count):
        new_row = []

        for row in range(row_count):
            new_row.append(matrix[row][column])

        transposed.append(new_row)

    return transposed

def matrix_multiply(matrix_a: list[list[float]], matrix_b: list[list[float]]) -> list[list[float]]:
    if not matrix_a or not matrix_b:
        raise ValueError("Matrices cannot be empty")

    matrix_a_columns = len(matrix_a[0])
    matrix_b_columns = len(matrix_b[0])

    if any(len(row) != matrix_a_columns for row in matrix_a):
        raise ValueError("Matrix A rows must have the same length")

    if any(len(row) != matrix_b_columns for row in matrix_b):
        raise ValueError("Matrix B rows must have the same length")

    if matrix_a_columns != len(matrix_b):
        raise ValueError("Matrix A columns must equal Matrix B rows")

    matrix_b_transposed = transpose(matrix_b)

    result = []

    for row in matrix_a:
        result_row = []

        for column in matrix_b_transposed:
            result_row.append(dot_product(row, column))

        result.append(result_row)

    return result

def matrix_add(matrix_a: list[list[float]], matrix_b: list[list[float]]) -> list[list[float]]:
    rows = len(matrix_a)
    columns = len(matrix_a[0])

    return [[matrix_a[row][column] + matrix_b[row][column] for column in range(columns)]for row in range(rows)]

if __name__ == "__main__":
    vector_a = [1, 2, 3]
    vector_b = [4, 5, 6]

    matrix_a = [
        [1, 2],
        [3, 4]
    ]

    matrix_b = [
        [5, 6],
        [7, 8]
    ]

    print("Dot Product:")
    print(dot_product(vector_a, vector_b))

    print("\nTranspose:")
    print(transpose(matrix_a))

    print("\nMatrix Multiplication:")
    print(matrix_multiply(matrix_a, matrix_b))