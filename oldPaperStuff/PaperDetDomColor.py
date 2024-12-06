import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

def segment_dominant_color(img,k=3):
    #reshape image into 2D array
    pixels = img.reshape((-1,3))
    pixels = np.float32(pixels)

    #define criteria for k-means clustering
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,10,1.0)

    #apply k-means clustering
    _,labels,centers = cv2.kmeans(pixels,k,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

    #convert labels back to original image
    labels = labels.reshape(img.shape[:2])

    #create mask for dominant color
    dominant_color_mask = np.where(labels == 0,255,0).astype(np.uint8)

    return dominant_color_mask

def domColor_paper(img):

    dominant_color_mask = segment_dominant_color(img) #find dominant color mask

    #morphological transformation
    kernel = np.ones((7,7),np.uint8)
    morph = cv2.morphologyEx(dominant_color_mask,cv2.MORPH_CLOSE, kernel=kernel)
    morph = cv2.morphologyEx(dominant_color_mask,cv2.MORPH_OPEN, kernel=kernel)

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

    polygon = img.copy()

    #overlay the new shape/polygon
    cv2.polylines(polygon,[corner],True,(255,0,0),1,cv2.LINE_AA)

    return polygon 
    