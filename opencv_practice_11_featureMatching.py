# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 09:29:36 2020

@author: Amirshayan
"""
import cv2
import numpy as np

img1 = cv2.imread('featureMatching2.jpg', 0)
img2 = cv2.imread('featureMatching.jpg', 0)


# Brutforce matching
# Detector of similarity
orb = cv2.ORB_create()
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

# Finding the key points
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)

matches = bf.match(des1, des2)
matches = sorted(matches, key = lambda x:x.distance)

img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:20], 
                       None, flags = 2)

cv2.imshow('Result', img3)
cv2.waitKey(0)
cv2.destroyAllWindows()
