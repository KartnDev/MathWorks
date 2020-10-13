import cv2
import numpy as np
from matplotlib import pyplot as plt

############# 1a

img = cv2.imread('Resources\\high_resolution.jpg')

#cv2.imshow('Original', img)
#cv2.waitKey(0)
for sized in [3, 5, 9, 11]:
    blur = cv2.GaussianBlur(img, (sized, sized), 0)

    #cv2.imshow(f'Blur {sized}x{sized}', blur)
    #cv2.waitKey(0)

############# 1b

blurEleven = cv2.GaussianBlur(img, (11, 11), 0)
blurFiveTwice = cv2.GaussianBlur(cv2.GaussianBlur(img, (5, 5), 0), (5, 5), 0)


numpy_vertical_concat = np.concatenate((blurEleven, blurFiveTwice), axis=0)
#cv2.imshow('5x5 powered twice Verse 11x11', numpy_vertical_concat)
#cv2.waitKey(0)

############# 2a

matrix = np.zeros((100, 100), float)
matrix[int(99/2)][int(99/2)] = 255

blur = cv2.GaussianBlur(matrix, (5, 5), 0)
#cv2.imshow('matrix 5x5 blur d', blur)
#cv2.waitKey(0)

############# 2b

blur = cv2.GaussianBlur(matrix, (9, 9), 0)
#cv2.imshow('matrix 9x9 blur d', blur)
#cv2.waitKey(0)

############# 2c

blurEleven = cv2.GaussianBlur(matrix, (9, 9), 0)
blurNineTwice = cv2.GaussianBlur(cv2.GaussianBlur(matrix, (5, 5), 0), (5, 5), 0)


numpy_vertical_concat = np.concatenate((blurEleven, blurNineTwice), axis=0)
#cv2.imshow('5x5 powered twice Verse 11x11', numpy_vertical_concat)
#cv2.waitKey(0)

############# 3a

for item in [1, 4, 9]:
    blur = cv2.GaussianBlur(img, (9, 9), sigmaX=item)
    #cv2.imshow(f'sigmaX={item} with size1=size2=9', blur)
    #cv2.waitKey(0)

############# 3b

for item in [1, 4, 9]:
    blur = cv2.GaussianBlur(img, (0, 0), sigmaX=item)
    #cv2.imshow(f'sigmaX={item} with size1=size2=0', blur)
    #cv2.waitKey(0)

############# 3c


blur = cv2.GaussianBlur(img, (0, 0), sigmaX=1, sigmaY=9)
#cv2.imshow(f'sigmaX=1 sigmaY=9 with size1=size2=0', blur)
#cv2.waitKey(0)

############# 3d

blur = cv2.GaussianBlur(img, (0, 0), sigmaX=9, sigmaY=1)
#cv2.imshow(f'sigmaX=9 sigmaY=1 with size1=size2=0', blur)
#cv2.waitKey(0)

############# 3e

blur = cv2.GaussianBlur(cv2.GaussianBlur(img, (0, 0), sigmaX=1, sigmaY=9), (0, 0), sigmaX=9, sigmaY=1)
#cv2.imshow('size1=size2=0 with sigmaX=1 and sigmaY=9 afterwards sigmaX=9 and sigmaY=1', blur)
#cv2.waitKey(0)

############# 3f

blur_first = cv2.GaussianBlur(img, (9, 9), sigmaX=0, sigmaY=0)
blur_second = cv2.GaussianBlur(cv2.GaussianBlur(img, (0, 0), sigmaX=1, sigmaY=9), (0, 0), sigmaX=9, sigmaY=1)


numpy_vertical_concat = np.concatenate((blur_first, blur_second), axis=0)

#cv2.imshow('sigmaX=0 sigmaY=0 with size1=size2=9 Verse'
#           ' size1=size2=0 with sigmaX=1 and sigmaY=9 afterwards sigmaX=9 and sigmaY=1', numpy_vertical_concat)
#cv2.waitKey(0)

############# 4a

src1 = cv2.imread('Resources\\ss1.png')
src2 = cv2.imread('Resources\\ss2.png')

diff12 = np.absolute(src2 - src1)

#cv2.imshow('diff12 image', diff12)
#cv2.waitKey(0)

############# 4b

kernel = np.ones((11, 11),np.uint8)
clean_diff = cv2.dilate(cv2.erode(diff12, kernel=kernel), kernel=kernel)

#cv2.imshow('clean diff image', clean_diff)
#cv2.waitKey(0)

############# 4c

dirty_diff = cv2.erode(cv2.dilate(diff12, kernel=kernel), kernel=kernel)

cv2.imshow('dirty diff image', dirty_diff)
cv2.waitKey(0)


