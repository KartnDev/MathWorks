import numpy as np


def full_circle_shift_clockwise(matrix_array: np.array, index_of_circle: int):
    not_shifted = matrix_array.copy()

    # init indexes
    last_index = 1

    # shift first row (init first and slice other)
    matrix_array[index_of_circle, index_of_circle] = not_shifted[index_of_circle + 1, index_of_circle]

    for i in range(index_of_circle + 1, matrix_array.shape[1] - index_of_circle):
        matrix_array[index_of_circle, i] = not_shifted[index_of_circle, i - 1]

    # shift last column
    for i in range(index_of_circle + 1, matrix_array.shape[0] - index_of_circle):
        matrix_array[i, matrix_array.shape[1] - index_of_circle - 1] = not_shifted[
            i - 1, matrix_array.shape[1] - index_of_circle - 1]

    # shift last row
    for i in reversed(range(index_of_circle, matrix_array.shape[1] - index_of_circle - 1)):
        matrix_array[matrix_array.shape[0] - index_of_circle - 1, i] = not_shifted[
            matrix_array.shape[0] - index_of_circle - 1, i + 1]

    for i in reversed(range(index_of_circle + 1, matrix_array.shape[0] - 1 - index_of_circle)):
        matrix_array[i, index_of_circle] = not_shifted[i + 1, index_of_circle]


def get_matrix_shift(matrix_array: np.array, shift_times: int = 1):
    result = matrix_array.copy()
    print(int(np.min(matrix_array.shape) / 2))

    for _ in range(0, shift_times):
        for shift_index in range(0, int(np.min(matrix_array.shape) / 2)):
            full_circle_shift_clockwise(result, shift_index)

    return result


if __name__ == '__main__':
    matrix = np.arange(0, 7 * 10).reshape(7, 10)
    print(matrix)
    print(get_matrix_shift(matrix))

