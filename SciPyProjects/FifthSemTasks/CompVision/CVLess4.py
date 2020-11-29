import cv2
import numpy as np
from matplotlib import pyplot as plt

# noinspection DuplicatedCode

#  12 (a)
img = cv2.imread('Resources\\high_resolution.jpg')
plt.imshow(img)
plt.show()

resized = img.copy()
for i in range(4):
    width = int(resized.shape[1] * 0.5)
    height = int(resized.shape[0] * 0.5)
    dim = (width, height)
    resized = cv2.resize(resized, dim, interpolation=cv2.INTER_AREA)

plt.imshow(resized)
plt.show()

# 12 B
new_resized = img.copy()
for i in range(4):
    width = int(new_resized.shape[1] * 0.5)
    height = int(new_resized.shape[0] * 0.5)
    dim = (width, height)
    new_resized = cv2.pyrDown(new_resized, dstsize=dim)

plt.imshow(new_resized)

# 13
ret, thresh = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)
ret, thresh = cv2.threshold(thresh, 128, 255, cv2.THRESH_BINARY_INV)
ret, thresh = cv2.threshold(thresh, 128, 255, cv2.THRESH_TRUNC)
ret, thresh = cv2.threshold(thresh, 128, 255, cv2.THRESH_TOZERO)
ret, thresh = cv2.threshold(thresh, 128, 255, cv2.THRESH_TOZERO_INV)
plt.imshow(thresh)
plt.show()

# 13 (A)
img = cv2.imread('Resources\\high_resolution.jpg', 0)
img = cv2.medianBlur(img, 5)

thresh = img.copy()
thresh = cv2.adaptiveThreshold(img, 128, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, C=5)
thresh = cv2.adaptiveThreshold(thresh, 128, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, C=5)

# noinspection DuplicatedCode
plt.imshow(thresh)
plt.show()

img = cv2.imread('Resources\\high_resolution.jpg', 0)
img = cv2.medianBlur(img, 5)

thresh = img.copy()
thresh = cv2.adaptiveThreshold(img, 128, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, C=5)
thresh = cv2.adaptiveThreshold(thresh, 128, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, C=5)

plt.imshow(thresh)
plt.show()

# 13 (B)
img = cv2.imread('Resources\\high_resolution.jpg', 0)
img = cv2.medianBlur(img, 5)
thresh = img.copy()
thresh = cv2.adaptiveThreshold(img, 128, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, C=0)
thresh = cv2.adaptiveThreshold(thresh, 128, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, C=0)
thresh = cv2.adaptiveThreshold(img, 128, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, C=-5)
thresh = cv2.adaptiveThreshold(thresh, 128, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, C=-5)
plt.imshow(thresh)
plt.show()

# 13 (C)
img1 = cv2.imread('Resources\\high_resolution.jpg', 0)
img1 = cv2.medianBlur(img, 5)
thresh = img.copy()
thresh = cv2.adaptiveThreshold(img1, 128, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, C=0)
thresh = cv2.adaptiveThreshold(thresh, 128, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, C=0)
thresh = cv2.adaptiveThreshold(img, 128, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, C=-5)
thresh = cv2.adaptiveThreshold(thresh, 128, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, C=-5)
plt.imshow(thresh)

img = cv2.imread('Resources\\high_resolution.jpg')
res = cv2.pyrMeanShiftFiltering(img, 70, 100)
plt.imshow(res)
plt.show()

img = cv2.imread('Resources\\circle.png')
plt.imshow(img)
plt.show()


# 15
kernel = np.array([[2, -1, -4],
                   [2, -4, -2],
                   [3.1, 3, -0.3]])

dst = cv2.filter2D(img, -1, kernel)
kernel = np.ones((5, 5), np.uint8)
dilated = cv2.dilate(dst, kernel, iterations = 1)
kernel = np.ones((7, 7), np.uint8)
opening = cv2.morphologyEx(dilated, cv2.MORPH_OPEN, kernel)
_,thresh = cv2.threshold(opening,10,255,cv2.THRESH_BINARY)
res = cv2.bitwise_and(img, cv2.bitwise_not(thresh))

plt.imshow(res)
plt.show()


# 16 (A)
img = cv2.imread('Resources\\high_resolution.jpg')
kernel = 1/16 * np.array([[1, 2, 1],
                          [2, 4, 2],
                          [1, 2, 1]])
dst = cv2.filter2D(img, -1, kernel, anchor=(0, 0))
plt.imshow(dst)
plt.show()

# 16 (B)

kernel1 =  1/4 * np.array([1, 2, 1])
kernel2 =  1/4 * np.array([[1],
                           [2],
                           [1]])
dst = cv2.filter2D(img, -1, kernel1, anchor=(0, 0))
dst = cv2.filter2D(dst, -1, kernel2, anchor=(0, 0))
plt.imshow(dst)
plt.show()


# 18
src = cv2.imread('Resources\\aim.png')

first_deriv = cv2.Laplacian(src, 0, ksize=3)
second_deriv = cv2.Laplacian(first_deriv, 0, ksize=3)

res = np.concatenate((first_deriv, second_deriv), axis=1) # 3x3
plt.imshow(res)
plt.show()


src = cv2.imread('Resources\\aim.png')

first_deriv = cv2.Laplacian(src, 0, ksize=5)
second_deriv = cv2.Laplacian(first_deriv, 0, ksize=5)

res = np.concatenate((first_deriv, second_deriv), axis=1) # 5x5
plt.imshow(res)
plt.show()


src = cv2.imread('Resources\\aim.png')

first_deriv = cv2.Laplacian(src, 0, ksize=9)
second_deriv = cv2.Laplacian(first_deriv, 0, ksize=9)

res = np.concatenate((first_deriv, second_deriv), axis=1) # 9x9
plt.imshow(res)
plt.show()


src = cv2.imread('Resources\\aim.png')

first_deriv = cv2.Laplacian(src, 0, ksize=9)
second_deriv = cv2.Laplacian(first_deriv, 0, ksize=9)

res = np.concatenate((first_deriv, second_deriv), axis=1) # 9x9
plt.imshow(res)
plt.show()