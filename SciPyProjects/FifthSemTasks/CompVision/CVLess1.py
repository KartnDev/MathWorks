import cv2
import matplotlib.pyplot as plt
import numpy as np

image = cv2.imread("../res/google.png")
src_mask = np.zeros(image.shape, image.dtype)

im_red, im_green, im_blue = cv2.split(image)

plt.imshow(im_green, cmap='gray')
plt.show()


clone1 = np.copy(im_green)
clone2 = np.copy(im_green)

cv2.imshow('Copy of Highway Picture', clone1)

thresh = (np.max(im_green) - np.min(im_green))

for i in range(clone1.shape[0]):
    for j in range(clone1.shape[1]):
        clone1[i][j] = thresh

for i in range(clone2.shape[0]):
    for j in range(clone2.shape[1]):
        clone2[i][j] = 0


cv2.compare(im_green, clone1, cv2.CMP_GE, clone2)

cv2.subtract(im_green, thresh/2, im_green, clone2)

plt.imshow(im_green, cmap='gray')
plt.show()