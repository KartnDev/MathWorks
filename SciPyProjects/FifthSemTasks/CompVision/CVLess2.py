import cv2 as cv
import numpy as np

A = [[26, 8, 4],
     [8, 26, 8],
     [4, 8, 26]]
# V1 = [[1, 0, 1], [0, 1, 0], [-1, 0, 1]]

E, U, V = cv.SVDecomp(np.float32(A))
E_improved = np.array([[E[0][0], 0, 0], [0, E[1][0], 0], [0, 0, E[2][0]]])
print(V)  # vectors
print(U)  # transposed vectors
print(E)
print(E_improved)
print()
# check
A_1 = U.dot(E_improved)
A_1 = A_1.dot(V)
print()
print(A_1)  # same
