import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

def threshold_paper(img):

    w,h = img.shape[:2]

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray,(75,75),0)

    thresh = cv2.threshold(blur,254,255,cv2.THRESH_BINARY & THRESH_OTSU)[1]

    