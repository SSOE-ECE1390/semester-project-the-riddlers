import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

#This function identifies the middle/center contour in an image using the identified contours and a specified point
def middle_contour(contours,point):
    min_dist = float('inf')
    closest_contour = None

    for contour in contours:

        dist = cv2.pointPolygonTest(contour,point,True)

        if abs(dist) < min_dist:
            min_dist = abs(dist)
            closest_contour = contour

    return closest_contour

def center_paper(img):

    #find image width and height
    w,h = img.shape[:2]

    #convert image to grayscale
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    #blur image, larger kernel provides closer fit than smaller kernel, but too big a kernel shrinks the paper
    blur = cv2.GaussianBlur(gray,(75,75),0)

    #threshold image for white
    thresh = cv2.threshold(blur,254,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]

    #morphological transformation
    kernel = np.ones((7,7),np.uint8)
    morph = cv2.morphologyEx(thresh,cv2.MORPH_CLOSE, kernel=kernel)
    morph = cv2.morphologyEx(morph,cv2.MORPH_OPEN, kernel=kernel)

    #find contours, outlines of blurred ROIs 
    contours = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  

    point = (np.round(w/2),np.round(h/2)) #identify center point of the image using width and height

    middle = middle_contour(contours,point) #use middle_contour function

    paper = np.zeros_like(img)

    #find contours, outlines of blurred ROIs 
    cv2.drawContours(paper,[middle], 0, (255,255,255),-1)

    peri = cv2.arcLength(middle,True) #parameter for approximation accuracy
    #approximate the paper outline with fewer indices, i.e. simplify a more complex shape to rectangle indices
    corner = cv2.approxPolyDP(middle,0.04*peri, True) 

    polygon = img.copy()

    #overlay the new shape/polygon
    cv2.polylines(polygon,[corner],True,(0,0,255),1,cv2.LINE_AA)

    return polygon