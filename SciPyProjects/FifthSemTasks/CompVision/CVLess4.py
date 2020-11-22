import cv2
import numpy as np

img = cv2.imread('Resources\\high_resolution.jpg')
# cv2.imshow('Original', img)
# cv2.waitKey(0)


#  12 (a)

resized = img.copy()

for i in range(4):
    width = int(resized.shape[1] * 0.5)
    height = int(resized.shape[0] * 0.5)
    dim = (width, height)
    resized = cv2.resize(resized, dim, interpolation=cv2.INTER_AREA)
# cv2.imshow('Original', resized)
# cv2.waitKey(0)

# 12 (b)

new_resized = img.copy()
for i in range(4):
    width = int(new_resized.shape[1] * 0.5)
    height = int(new_resized.shape[0] * 0.5)
    dim = (width, height)
    new_resized = cv2.pyrDown(new_resized, dstsize=dim)

# cv2.imshow('Original', new_resized)
# cv2.waitKey(0)

# 13 (a)


res = cv2.pyrMeanShiftFiltering(img, 10, 100)

cv2.imshow('Original', res)
cv2.waitKey(0)

img = cv2.imread('Resources\\high_resolution.jpg')

kernel = np.array([[0, 0, 0, 0, 0, 0, 1, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 1, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 1, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 1, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 1, 0, 0, 0, 0, 0, 0]])


