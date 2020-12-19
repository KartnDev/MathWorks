import cv2
from PIL import ImageDraw, Image, ImageStat
import matplotlib.pyplot as plt


def match_img_dist(img1, img2):
    orb = cv2.ORB()

    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    matches = bf.match(des1, des2)

    matches = sorted(matches, key=lambda x: x.distance)

    return matches


img = cv2.imread('Resources//index.png', 0)
scaled = cv2.resize(img, (0, 0), fx=0.2, fy=0.2)

print(match_img_dist(img, scaled))