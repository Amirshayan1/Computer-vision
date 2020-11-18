# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 10:42:39 2020

@author: Amirshayan
"""
# This method considers forground as moving object and not moving 
# as Background
import cv2
import numpy as np

cap = cv2.VideoCapture(0)
fgbg = cv2.createBackgroundSubtractorMOG2()

while True:
    
    ret, frame = cap.read()
    fgmask = fgbg.apply(frame)
    
    cv2.imshow('original', frame)
    cv2.imshow('fg', fgmask)
    
    k = cv2.waitKey(30) & 0xFF
    if k == 27:
        break
    
cap.release()
cv2.destroyAllWindows()