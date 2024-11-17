import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

def threshold_paper(img):

    w,h = img.shape[:2] #find image width and height

    #convert image to grayscale
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    #blur image, larger kernel provides closer fit than smaller kernel, but too big a kernel shrinks paper edges
    blur = cv2.GaussianBlur(gray,(75,75),0)

    #threshold image for white
    thresh = cv2.threshold(blur,254,255,cv2.THRESH_BINARY & THRESH_OTSU)[1]

    #morphological transformations
    kernel = np.ones((7,7),np.uint8)
    morph = cv2.morphologyEx(thresh,cv2.MORPH_CLOSE, kernel=kernel)
    morph = cv2.morphologyEx(morph,cv2.MORPH_OPEN, kernel=kernel)

    #find contours, outlines of blurred ROIs
    contours = cv2.findContours(morph, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    
    contours = contours[0] if len(contours) == 2 else contours[1]
    area_thresh = 0

    for c in contours:
        area = cv2.contourArea(c) #find area of the contour

        if area > area_thresh:
            area_thresh = area
            contourMax = c

    paper = np.zeros_like(img)

    #draw contour lines on the image using coordinates found in largest contour area
    cv2.drawContours(paper,[contourMax],0,[255,255,255],-1)

    peri = cv2.arcLength(contourMax,True) #parameter for approx accuracy
    corner = cv2.approxPolyDP(contourMax,0.04*peri,True) #approx paper outline with fewer indices, i.e. simplify more complex shape

    polygon = img.copy()

    #overlay new shape/polygon
    cv2.polylines(polygon,[corner],True,(255,0,0),1,cv2.LINE_AA)

    return polygon 

    