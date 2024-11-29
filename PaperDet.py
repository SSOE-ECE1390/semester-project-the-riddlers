import cv2
import os
import numpy as np
import matplotlib.pyplot as plt


def threshold_paper(contours):

        #image preprocessing 
    img = cv2.imread('testImage.png') 

    #find image width and height
    w,h = img.shape[:2]
    B,G,R = cv2.split(img)

    #convert image to grayscale
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    th,thresh1_img = cv2.threshold(B,170,255,cv2.THRESH_BINARY)
    th,thresh2_img = cv2.threshold(G,170,255,cv2.THRESH_BINARY)
    th,thresh3_img = cv2.threshold(R,170,255,cv2.THRESH_BINARY)

    anded = cv2.bitwise_and(thresh1_img, thresh2_img, thresh3_img)

    kernel = np.ones((5,5),np.uint8)
    erosion = cv2.erode(anded,kernel,iterations=8) 
    dilation = cv2.dilate(erosion,kernel,iterations=15)

    #blur image, larger kernel provides closer fit than smaller kernel, but too big a kernel shrinks the paper
    blur = cv2.GaussianBlur(dilation,(55,55),0)

    #threshold image for white
    thresh = cv2.threshold(blur,25,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]

    #morphological transformation
    kernel = np.ones((7,7),np.uint8)
    morph = cv2.morphologyEx(thresh,cv2.MORPH_CLOSE, kernel=kernel)
    morph = cv2.morphologyEx(morph,cv2.MORPH_OPEN, kernel=kernel)

    #find contours, outlines of blurred ROIs 
    contours = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contours = contours[0] if len(contours) == 2 else contours[1]
    area_thresh = 0

    for c in contours:
        area = cv2.contourArea(c) #find area of the contour
        if area > area_thresh:
            area_thresh = area
            contourMax = c #identify largest contour area -> this should be your paper

    paper = np.zeros_like(img)

    #draw contour lines on the image using coordinates found in largest contour area
    cv2.drawContours(paper,[contourMax], 0, (255,255,255),-1) 

    peri = cv2.arcLength(contourMax,True) #parameter for approximation accuracy
    #approximate the paper outline with fewer indices, i.e. simplify a more complex shape to rectangle indices
    corner = cv2.approxPolyDP(contourMax,0.04*peri, True)

    x,y,w,h = cv2.boundingRect(contourMax) #find rectangle that bounds the contour
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

    rect2 = cv2.minAreaRect(contourMax) #find smallest possible rectangle to bind contour
    box = cv2.boxPoints(rect2)
    box = np.int0(box)
    img = cv2.drawContours(img,[box],0,(0,0,255),2)

    polygon = img.copy()

    #overlay the new shape/polygon
    cv2.polylines(polygon,[corner],True,(255,0,0),5,cv2.LINE_AA)

    return polygon